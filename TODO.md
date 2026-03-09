# nanobot TODO

计划中的功能特性与具体修改方案。所有改动遵循 nanobot 轻量风格，对已有数据向下兼容。

---

## 1. `/drop` 命令 —— 撤销上一轮对话

### 功能说明

与 `/stop` 不同，`/drop` 不仅停止当前任务，还会：

1. **删除上一轮对话产生的所有 session 记录**（用户消息、assistant 回复、tool_calls、tool results），防止上下文污染。
2. **在支持撤回/删除的渠道（如 Telegram）上，删除该轮对话中 bot 发出的所有可见消息**（包括 progress 消息、exec 结果等）。

### 现有代码分析

| 组件 | 文件 | 现状 |
|------|------|------|
| 命令分发 | `agent/loop.py:271-276` | `/stop` 在主消息循环中拦截处理，`/new`、`/help` 在 `_process_message()` 中处理 |
| 任务取消 | `agent/loop.py:278-292` | `_handle_stop()` 取消 asyncio tasks + subagents，但不触碰 session 数据 |
| 消息存储 | `agent/loop.py:455-488` | `_save_turn()` 向 `session.messages` 追加当前轮次产生的所有消息 |
| Session 类 | `session/manager.py:16-70` | append-only 设计，只有 `clear()` 清空全部消息，无删除末尾消息的方法 |
| Telegram 发送 | `channels/telegram.py:295-363` | 发送消息但不追踪已发送的 `message_id`，无 `delete_message` 调用 |

### 具体修改方案

#### 1.1 Session 新增 `drop_last_turn()` 方法

**文件**: `session/manager.py`

在 `Session` 类中新增方法，从 `messages` 列表末尾反向查找最近一个 `user` 角色消息的位置，然后截断该位置之后的所有消息（即删除最后一轮 user→assistant→tool 交互）。

```python
def drop_last_turn(self) -> int:
    """Remove the last conversation turn (from last user message onwards).
    Returns the number of messages removed."""
    # 反向查找最后一个 user 消息（datetime 已在文件顶部导入）
    for i in range(len(self.messages) - 1, -1, -1):
        if self.messages[i].get("role") == "user":
            removed = len(self.messages) - i
            self.messages = self.messages[:i]
            self.last_consolidated = min(self.last_consolidated, len(self.messages))
            self.updated_at = datetime.now()
            return removed
    return 0
```

此方法只删除尾部未归档的对话。`last_consolidated` 通过 `min()` 自动修正，避免越界。

#### 1.2 Agent Loop 添加 `/drop` 命令处理

**文件**: `agent/loop.py`

在主消息循环 `run()` 方法中（紧邻 `/stop` 检查），添加 `/drop` 分支：

```python
# 在 line 271 附近
if msg.content.strip().lower() == "/stop":
    await self._handle_stop(msg)
elif msg.content.strip().lower() == "/drop":
    await self._handle_drop(msg)
else:
    ...
```

新增 `_handle_drop()` 方法：

1. 先调用 `_handle_stop()` 逻辑取消正在运行的任务
2. 获取 session，调用 `session.drop_last_turn()`
3. 保存 session
4. 通知渠道删除已发送的消息（如适用）
5. 发送确认消息

#### 1.3 渠道消息追踪与删除

**文件**: `channels/base.py` — 在 `BaseChannel` 中新增可选 `delete(chat_id, message_id)` 方法（默认空实现，不破坏现有渠道）。

**文件**: `channels/telegram.py` — 实现 Telegram 消息删除：

1. **追踪已发送消息**: 在 `send()` 方法中，记录每次 `send_message` / `send_photo` 等返回的 `message_id` 到一个 per-session 列表（如 `_sent_message_ids: dict[str, list[int]]`）。
2. **实现删除**: 新增 `delete_messages(chat_id, message_ids)` 方法，调用 `self._app.bot.delete_message(chat_id, msg_id)`。
3. **`/drop` 集成**: `_handle_drop()` 中通过 bus 发送特殊 `OutboundMessage`（带 metadata 标记 `_action: "delete"`），Telegram 渠道识别此标记后执行批量删除。

#### 1.4 `/help` 更新

**文件**: `agent/loop.py:392-394` — 在 help 文本中新增 `/drop` 说明。

**文件**: `channels/telegram.py` — 在 `BotCommand` 列表中注册 `/drop`。

#### 向下兼容

- `drop_last_turn()` 仅操作内存中的 `messages` 列表，JSONL 格式不变
- 新字段均可选，旧数据加载不受影响
- 不支持删除的渠道（Slack、Discord 等）无需改动，`/drop` 仅清除 session 记录

---

## 2. 模型热切换

### 功能说明

允许用户在对话中通过命令（如 `/model deepseek/deepseek-chat`）切换当前会话使用的模型，而非全局固定一个模型。

### 现有代码分析

| 组件 | 文件 | 现状 |
|------|------|------|
| 模型配置 | `config/schema.py:221-233` | `AgentDefaults.model` 全局单一模型 |
| AgentLoop 初始化 | `agent/loop.py:74` | `self.model = model or provider.get_default_model()` 启动时固定 |
| LLM 调用 | `agent/loop.py` (_run_agent_loop) | 每次调用 `provider.chat(model=self.model, ...)` |
| Provider 匹配 | `config/schema.py:346-388` | `_match_provider(model)` 支持按模型名自动匹配 provider |
| Session | `session/manager.py` | `metadata: dict` 字段可自由扩展 |

### 具体修改方案

#### 2.1 Session 级别模型覆盖

**文件**: `session/manager.py`

利用现有 `Session.metadata` 字段存储模型覆盖：

```python
# 无需改动 Session 类，直接使用 metadata
session.metadata["model_override"] = "deepseek/deepseek-chat"
```

#### 2.2 Agent Loop 获取有效模型

**文件**: `agent/loop.py`

新增辅助方法：

```python
def _effective_model(self, session: Session) -> str:
    """Return session-level model override, or fall back to default."""
    return session.metadata.get("model_override") or self.model
```

在 `_process_message()` 和 `_run_agent_loop()` 中，将 `self.model` 替换为 `self._effective_model(session)`。

#### 2.3 `/model` 命令

**文件**: `agent/loop.py`

在 slash 命令区域新增（`cmd` 已在上方定义为 `msg.content.strip().lower()`）：

```python
if cmd.startswith("/model"):
    parts = msg.content.strip().split(maxsplit=1)
    if len(parts) < 2:
        current = session.metadata.get("model_override") or self.model
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content=f"Current model: {current}",
        )
    new_model = parts[1].strip()
    session.metadata["model_override"] = new_model
    self.sessions.save(session)
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id,
        content=f"Model switched to: {new_model}",
    )
```

#### 2.4 Provider 自动匹配

已有 `_match_provider(model)` 逻辑支持按模型名自动匹配 provider，无需额外改动。当 session 级别 model 改变时，provider 会自动匹配。

#### 向下兼容

- 使用现有 `session.metadata` 字段，不引入新的数据结构
- 旧 session 无 `model_override` 键，`dict.get()` 返回 `None`，回退到全局默认模型
- JSONL 格式不变，metadata 已序列化在首行

---

## 3. SubAgent 自定义模型参数

### 功能说明

允许主 agent 在 spawn subagent 时指定使用不同于默认模型的模型，例如用低成本模型执行简单任务。

### 现有代码分析

| 组件 | 文件 | 现状 |
|------|------|------|
| SpawnTool | `agent/tools/spawn.py:39-53` | 参数仅有 `task` 和 `label`，无 `model` |
| SubagentManager.spawn() | `agent/subagent.py:53-83` | 接受 `task`, `label` 等，使用全局 `self.model` |
| _run_subagent() | `agent/subagent.py:85-178` | `provider.chat(model=self.model, ...)` 固定使用全局模型 |

### 具体修改方案

#### 3.1 SpawnTool 新增 `model` 参数

**文件**: `agent/tools/spawn.py`

在 `parameters` 属性中新增：

```python
"model": {
    "type": "string",
    "description": "Optional: LLM model for the subagent (defaults to the current model)",
},
```

在 `execute()` 中传递：

```python
async def execute(self, task: str, label: str | None = None, model: str | None = None, **kwargs) -> str:
    return await self._manager.spawn(
        task=task, label=label, model=model,
        origin_channel=self._origin_channel,
        origin_chat_id=self._origin_chat_id,
        session_key=self._session_key,
    )
```

#### 3.2 SubagentManager.spawn() 传递 model

**文件**: `agent/subagent.py`

`spawn()` 方法新增 `model` 参数，传递给 `_run_subagent()`：

```python
async def spawn(self, task, label=None, model=None, ...) -> str:
    bg_task = asyncio.create_task(
        self._run_subagent(task_id, task, display_label, origin, model=model)
    )
```

#### 3.3 _run_subagent() 使用指定模型

**文件**: `agent/subagent.py`

`_run_subagent()` 新增 `model` 参数，在 `provider.chat()` 调用中使用：

```python
async def _run_subagent(self, task_id, task, label, origin, model=None):
    effective_model = model or self.model
    ...
    response = await self.provider.chat(
        messages=messages,
        model=effective_model,  # 使用指定模型
        ...
    )
```

#### 向下兼容

- `model` 参数可选，默认 `None` 回退到全局模型
- 旧版 config 无需改动
- LLM 不传 `model` 参数时行为不变

---

## 修改文件清单

| 文件 | 变更类型 | 涉及功能 |
|------|---------|---------|
| `session/manager.py` | 新增方法 | `/drop` |
| `agent/loop.py` | 新增命令处理 | `/drop`, `/model` |
| `agent/subagent.py` | 方法签名扩展 | SubAgent 模型 |
| `agent/tools/spawn.py` | 参数扩展 | SubAgent 模型 |
| `channels/base.py` | 新增可选方法 | `/drop` 渠道删除 |
| `channels/telegram.py` | 消息追踪+删除 | `/drop` Telegram |

**预估新增代码**: ~100-150 行（保持轻量风格）
