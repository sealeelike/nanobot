# TODO / 设计备忘

本文档只记录现有代码调研结果与后续修改建议，**本次不改运行时行为**。目标是为 `/drop`、`/model` 和 subagent 额外模型配置提供一个尽量轻量、易维护、对旧数据无缝兼容的实现方案。

## 为什么新增 TODO 文件，而不是继续扩写 README

- `README.md` 已经很长，适合放用户视角的安装、配置、功能介绍。
- 这次需要记录的是**面向开发实现的具体改动建议**，独立文件更容易逐项落地，也更符合 nanobot 保持轻量、直观的风格。
- 后续如果真正开工，可以直接按本文件列出的“文件级改动清单”和“测试清单”分步提交。

## 现状调研

### 1. 用户命令入口

- `nanobot/agent/loop.py`
  - `run()` 里目前只直接拦截 `/stop`
  - `_process_message()` 里目前处理 `/new`、`/help`
- `nanobot/channels/telegram.py`
  - Telegram 自己也有一份 `/help` 文案
  - 其他命令会统一转发给 `AgentLoop`

### 2. 会话与持久化

- `nanobot/session/manager.py`
  - 会话保存在 `workspace/sessions/*.jsonl`
  - 第一行是 metadata，后续每行是一条 message
  - 已经有旧版 `~/.nanobot/sessions/` 自动迁移逻辑
- `Session.messages` 当前是**追加式**历史，`last_consolidated` 用来标记哪些消息已归档进 `MEMORY.md` / `HISTORY.md`
- `AgentLoop._save_turn()` 会把本轮新增的 user / assistant / tool 消息追加到 `session.messages`

### 3. 模型选择

- `nanobot/config/schema.py`
  - 全局默认模型来自 `agents.defaults.model`
  - `Config.get_provider_name(model)` / `get_provider(model)` 根据模型名和 provider 配置做匹配
- `nanobot/cli/commands.py`
  - `_make_provider(config)` 只在启动时按**一个全局 model** 构造 provider
- `nanobot/agent/loop.py`
  - `self.model` 与 `self.provider` 在初始化后固定

### 4. Subagent

- `nanobot/agent/subagent.py`
  - subagent 默认继承主 agent 的 provider / model
  - 当前没有单独模型配置，也没有“允许使用额外模型”的参数
- `nanobot/agent/tools/spawn.py`
  - `spawn` 只有 `task` / `label` 参数，没有 `model`

### 5. 渠道前端消息删除能力

- `nanobot/channels/base.py`
  - 只有 `send()` 抽象，没有删除/撤回抽象
- `nanobot/channels/telegram.py`
  - 发送文本时能拿到 Telegram API 返回对象的能力是有的，但当前没有保存“本轮发送了哪些 message_id”，也没有删除逻辑
- 目前 `OutboundMessage.metadata` 已经承载了 `_progress`、`message_id` 等路由信息，说明**继续走 metadata 扩展**会比大改消息总线结构更轻

---

## 建议一：新增 `/drop`，撤销上一轮对话并清除记录

### 目标

`/drop` 不只是“停掉正在运行的任务”，而是：

1. 取消当前 session 上一轮仍在执行的任务 / subagent；
2. 从会话历史中删除“上一轮”产生的 user / assistant / tool / system 记录，避免污染后续上下文；
3. 对支持撤回/删除的渠道（至少 Telegram），尽量把上一轮产生的前端可见消息一并删除；
4. 保持对旧 session JSONL 的兼容，老数据无需迁移脚本即可继续用。

### 最小改动方案

#### A. 给“轮次”补一个轻量标记，而不是重做 session 格式

建议在 `session.messages` 的每条新消息里增加**可选字段**：

- `turn_id`: 当前轮次唯一标识
- `turn_role`: 可选，标记 `user_input` / `assistant_output` / `tool_result` / `system_result`

这样做的优点：

- JSONL 顶层结构不变，旧代码/旧数据都不会被破坏；
- 旧 session 没有 `turn_id` 也没关系，可以回退到“从尾部向前找最后一个 user 消息作为最近一轮起点”的兼容逻辑；
- 仍然保留 append-only 的主结构，符合 nanobot 当前轻量实现风格。

#### B. 明确“上一轮”的边界

建议在 `AgentLoop._process_message()` 为每个进入正常 LLM 流程的用户消息生成一个 `turn_id`，并把它透传到：

- `_save_turn()` 写入的 session message；
- 本轮对外发送的 progress / exec / 最终回复；
- 本轮 spawn 的 subagent；
- subagent 回传结果对应的 system message。

这样 `/drop` 只需要删除最后一个 `turn_id` 对应的记录，不需要猜太多上下文。

#### C. `/drop` 行为应建立在 `/stop` 之上，但不能只做 `/stop`

建议在 `nanobot/agent/loop.py` 增加 `_handle_drop()`：

1. 先复用当前 `/stop` 的取消逻辑，确保未完成任务不会继续写回历史；
2. 找出 session 最后一个完整 `turn_id`（或兼容旧数据时回退为“最后一个 user 起点到结尾”）；
3. 从 `session.messages` 中删除这一轮的所有记录；
4. 清理与该轮相关的附件索引/前端消息回执；
5. `self.sessions.save(session)` 持久化；
6. 回发一条简洁确认消息，例如 `🗑 Dropped the previous turn.`

#### D. 前端消息删除不要强依赖重构总线，优先走 metadata 扩展

为了保持轻量，建议**不要**先把 `BaseChannel` 全面升级成复杂的“消息操作总线”。先走当前已有的 `OutboundMessage.metadata` 扩展：

- 本轮正常发送时，在 metadata 带上 `turn_id`
- 支持删除的渠道在发送成功后，记录 `(chat_id, turn_id) -> [message_id, ...]`
- `/drop` 时发一条特殊 outbound，metadata 中带：
  - `_delete_turn`: `<turn_id>`
  - `_silent`: `True`

然后：

- `ChannelManager` 仍然按 channel 转发；
- `TelegramChannel.send()` 识别 `_delete_turn` 后，调用 Telegram delete API 删除该轮的已发送消息；
- 不支持删除的渠道忽略该 metadata，保持兼容。

这条路径比“新增 delete()/edit()/receipt 抽象并改所有渠道”更小，也更符合 nanobot 的轻量定位。

#### E. 回执存储建议：只存最近几轮，放在 session.metadata 里

建议给 `Session.metadata` 增加**可选**结构，例如：

```json
{
  "turn_receipts": {
    "turn-123": {
      "telegram": ["101", "102", "103"]
    }
  }
}
```

约束：

- 只保留最近几轮（例如 5~10 轮）；
- 不要求老 session 有这个字段；
- 字段缺失时，`/drop` 只删历史，不做前端删除；
- 这样即使 agent 重启，最近一轮的 Telegram 消息仍有机会被删除，不必只依赖内存缓存。

#### F. 与 memory consolidation 的兼容原则

`/drop` 针对的是“上一轮”，因此建议未来实现时显式保证：

- **最新一轮完整对话永远不要被归档进 `MEMORY.md` / `HISTORY.md`**

也就是在 `MemoryStore.consolidate()` 的切片逻辑上，额外确保“最近完整一轮”仍留在 `session.messages` 中。这样 `/drop` 不需要倒推重写归档文件，能大幅降低复杂度。

### 建议涉及文件

- `nanobot/agent/loop.py`
- `nanobot/session/manager.py`
- `nanobot/agent/memory.py`
- `nanobot/channels/manager.py`
- `nanobot/channels/telegram.py`
- `nanobot/channels/base.py`（如最终需要补最小能力说明，可选）

---

## 建议二：新增 `/model`，支持会话级热切换

### 目标

当前 `agents.defaults.model` 是全局唯一模型。建议新增会话级 `/model`：

- `/model`：查看当前 session 正在使用的模型，以及全局默认模型；
- `/model <model-name>`：切换当前 session 的活动模型；
- `/model default`：清除会话覆盖，回退到全局默认。

### 最小改动方案

#### A. 不直接改全局 config，改成 session 级 override

建议新增一个**可选** session metadata 字段：

- `session.metadata["active_model"]`

读取顺序：

1. 如果 session metadata 里有 `active_model`，优先使用；
2. 否则回退到 `agents.defaults.model`。

这样：

- 老 session 没有该字段时，行为与现在完全一致；
- 不需要修改现有 config 文件格式即可兼容旧版本；
- `/model` 的影响范围仅限当前会话，不会污染其他聊天渠道/群/线程。

#### B. 把“按模型拿 provider”的能力从 CLI 启动阶段提出来

当前 `_make_provider(config)` 在启动时只生成一个固定 provider，这不适合热切换。

建议后续改成一个轻量 resolver/factory，例如：

- `cli/commands.py`：把 `_make_provider(config)` 拆成“按指定 model 创建 provider”的函数
- `AgentLoop`：接收一个 `provider_factory(model) -> LLMProvider`

运行时每轮处理消息时：

1. 先解析当前 session 的 active model；
2. 再用 resolver 创建对应 provider；
3. 用该 provider + model 执行本轮。

优点：

- 逻辑仍然集中在现有 config/provider 匹配体系，不另起一套状态机；
- 不需要把全局 `Config` 频繁写回磁盘；
- 以后 subagent 复用同一 resolver 即可。

#### C. `/model` 命令尽量保持简单

建议放在 `AgentLoop._process_message()` 里，与 `/new`、`/help` 同级处理。

推荐返回文案：

- 查看：
  - `Current model: xxx`
  - `Default model: yyy`
- 切换成功：
  - `Switched this session to xxx.`
- 回退成功：
  - `Reverted this session to the default model.`
- 校验失败：
  - `Model xxx is not available with current provider/API key configuration.`

### 建议涉及文件

- `nanobot/cli/commands.py`
- `nanobot/agent/loop.py`
- `nanobot/config/schema.py`
- `nanobot/session/manager.py`
- `nanobot/channels/telegram.py`（同步补 `/help` 文案）

---

## 建议三：给 subagent 增加“额外可用模型”配置

### 目标

这不是前端命令，而是后端配置能力：

- 主 agent 仍然有一个默认模型；
- subagent 除了继承 default model，还可以在配置中声明若干“允许额外使用”的模型；
- `spawn` 工具可以在内部指定 model，但只能选默认模型或白名单模型，避免失控。

### 最小改动方案

#### A. 配置层新增可选 subagent 小节

建议在 `nanobot/config/schema.py` 增加：

- `agents.subagent.model: str | None = None`
  - 可选，给 subagent 一个默认覆盖模型
- `agents.subagent.extra_models: list[str] = []`
  - 可选，允许 subagent 使用的额外模型白名单

兼容性：

- 旧 config 没有 `subagent` 时，Pydantic 默认值兜底；
- 不需要写迁移脚本；
- 这与现有 `agents.defaults.*` 保持相同风格。

#### B. `spawn` 工具新增可选 `model` 参数，但只对内部生效

建议在 `nanobot/agent/tools/spawn.py` 增加可选参数：

- `model: str | None = None`

约束：

- 若未传，按 `subagent.model -> 主 session active_model -> agents.defaults.model` 顺序选择；
- 若传了，必须在 `{默认模型, subagent.model, extra_models}` 白名单里，否则拒绝；
- 这样不会把“任意模型切换”开放成不受控入口。

#### C. SubagentManager 改为复用与主 agent 相同的 model resolver

建议 `SubagentManager` 不再只保存一个固定 `provider + model`，而是保存：

- 当前配置下的 allowed models
- 与主 agent 共用的 provider factory / model resolver

这样 subagent 可以在不复制 provider 匹配逻辑的情况下安全切模型，仍然保持代码薄。

### 建议涉及文件

- `nanobot/config/schema.py`
- `nanobot/agent/subagent.py`
- `nanobot/agent/tools/spawn.py`
- `nanobot/agent/loop.py`

---

## 向下兼容要求

未来正式实现时，建议严格守住以下约束：

1. **session JSONL 顶层格式不变**
   - 仍然是一行 metadata + 多行 message
   - 只新增可选字段，不改旧字段含义

2. **旧 session 无需迁移**
   - 没有 `turn_id` 时，`/drop` 使用回退算法识别上一轮
   - 没有 `active_model` 时，继续使用全局默认模型
   - 没有 `turn_receipts` 时，不做前端删除，仅删除上下文历史

3. **旧 config 无需迁移**
   - 没有 `agents.subagent` 时使用默认值
   - 现有 `agents.defaults.model` 仍然是全局回退项

4. **保留现有 legacy session 迁移逻辑**
   - 不要破坏 `~/.nanobot/sessions/ -> workspace/sessions/` 的自动迁移

5. **不要引入重型抽象**
   - 优先复用 `Session.metadata`
   - 优先复用 `OutboundMessage.metadata`
   - 优先在现有 `AgentLoop` / `SubagentManager` 上做增量修改

---

## 推荐测试清单（真正实现时再补）

### `/drop`

- 旧格式 session（无 `turn_id`）仍能删除最后一轮
- 新格式 session（有 `turn_id`）能精确删除最后一轮
- `/drop` 会先取消活跃 task/subagent，再删历史
- 删除后不会残留该轮 tool/system 消息污染下一轮上下文
- Telegram 渠道在有 `turn_receipts` 时会触发删除逻辑

### `/model`

- `/model` 无参数时能正确展示当前/默认模型
- `/model <name>` 只影响当前 session
- `/model default` 能正确回退
- 配置不足（无 API key / provider 不可解析）时返回清晰错误
- 旧 session 没有 `active_model` 时行为不变

### subagent 额外模型

- 未配置 `agents.subagent` 时仍沿用当前默认行为
- 配置 `subagent.model` 后 spawn 默认使用该模型
- 配置 `extra_models` 后，白名单内模型可用、白名单外模型被拒绝
- subagent 与主 agent 共用 provider 解析逻辑，不出现一边能跑一边不能跑的分叉行为

---

## 推荐的实际改动顺序

1. 先抽出“按 model 解析 provider”的公共能力；
2. 再落地 session 级 `active_model` + `/model`；
3. 再给消息保存补 `turn_id`；
4. 基于 `turn_id` 实现 `/drop` 删除历史；
5. 最后给 Telegram 增加前端消息删除回执与实际删除逻辑；
6. 再扩展 subagent 的 `model` / `extra_models` 能力。

这样可以最大限度降低回归风险，也更容易保证每一步都保持 nanobot 现有的轻量风格。
