---

layout: post
tags: [LLM, Agent, Engineering]
title: Claude Code 如何工作
date: 2026-04-01
author: Kyrie Chen
comments: true
toc: true
pinned: false

---

最近系统性梳理了一遍 Claude Code 的整套设计，发现它和很多"看起来很像魔法"的 agent 产品不太一样。它确实是一个 terminal-based coding agent，但真正支撑它稳定运行的，不是某种特别花哨的推理框架，而是一套很扎实的工程约束：上下文怎么组装，memory 怎么分层，tool 怎么暴露，权限怎么裁剪，配置怎么覆盖，长对话怎么继续跑。

一开始我只是想把 Claude Code 的运行机制讲明白，后来发现只讲 agentic loop 远远不够。真正决定它体验上限的，其实是另一整套配套机制：`Core Concepts` 里的上下文、权限、tools，`Guides` 里的认证、hooks、MCP、skills，以及 `Configuration` 里的分层配置和环境变量。于是这篇文章干脆把这些部分放在一起，整理成一篇更完整的中文长文。


---

> 这篇文章按实际使用顺序来组织：先讲 `Get Started`，再讲 `Core Concepts`，最后进入 `Guides` 和 `Configuration`。这样读下来会更像一次完整上手，而不是零散知识点的拼接。

## Get Started

把 `Get Started` 放在最前面是合理的。Claude Code 不是那种先读完架构图再开始用的产品，它更像一个需要先跑起来、再逐步理解内部机制的终端工具。

### Introduction：它首先是一个 terminal-native 的 coding agent

Claude Code 的定位可以概括得很直接：它不是一个"带聊天框的 IDE 插件"，而是一个直接跑在终端里的 agent，默认就能碰到你的代码库、shell 和本地工具链。

它开箱即用的能力大致分成六类：

| 能力 | 典型动作 |
| --- | --- |
| **Read & edit files** | 读取源码、做定点修改、生成 diff |
| **Run shell commands** | 跑测试、构建、git、脚本 |
| **Search your codebase** | 按文件名、glob、正则快速定位 |
| **Fetch from the web** | 抓文档页、查 API、做网页搜索 |
| **Spawn sub-agents** | 把复杂任务拆成并行工作流 |
| **Connect MCP servers** | 接数据库、浏览器、外部 API、内部系统 |

和很多 agent 产品相比，Claude Code 最不一样的一点是：它不要求你先切到某个专有 UI。终端就是一等运行环境，所以后面的 memory、permissions、hooks、settings 都围绕"如何在真实开发环境中安全运行"来设计。

### Quickstart：官方的最短路径其实只有三步

Quickstart 的核心其实就三件事：

1. 安装 Claude Code
2. 完成认证
3. 在项目目录里直接开工

```bash
npm install -g @anthropic-ai/claude-code
claude

cd my-project
claude "add error handling to the API client"
```

如果你已经在交互 session 里，官方非常建议立刻跑一次：

```text
/init
```

它会扫描当前仓库，生成一份项目级 `CLAUDE.md` 初稿。这个动作很关键，因为 Claude Code 真正稳定的体验，往往不是从第一次 prompt 开始的，而是从你把项目规则明确写进 memory 开始的。

### Installation：安装页真正有价值的是"边界条件"

安装页不只是给一条 `npm install` 命令，它真正补的是几个容易踩坑的边界条件。


**系统要求**

| 项目 | 要求 |
| --- | --- |
| **OS** | macOS 10.15+、Ubuntu 20.04+/Debian 10+、Windows 10+（WSL / WSL2 / Git for Windows） |
| **Node.js** | 18+ |
| **内存** | 4GB+ |
| **网络** | 认证和模型调用都需要联网 |
| **Shell** | Bash / Zsh / Fish 体验最好 |

额外依赖里最值得记的是 `ripgrep`。很多搜索能力虽然通常会随安装一起带上，但一旦你的搜索功能异常，第一反应就该检查 `rg` 是否真的在 PATH 里。


**标准安装 vs 原生二进制安装**

最传统的安装方式依然是：

```bash
npm install -g @anthropic-ai/claude-code
```

现在值得注意的是，Claude Code 已经不再只是一包 npm JS，它在逐步演进到更接近原生 CLI 分发的形态。常见入口有：

```bash
brew install --cask claude-code
```

以及：

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

这背后的文档信号很明确：Claude Code 正在从"开发者自己愿意折腾一下的 npm 工具"，向"更像正式桌面/CLI 产品的分发方式"过渡。


**安装页里最容易忽略的两个提醒**

- **不要用** `sudo npm install -g @anthropic-ai/claude-code`。这会把权限和后续升级都搞得很脆弱。
- 如果你走 Homebrew 或安装脚本的原生分发，**自动更新** 就成了运行时的一部分；想关闭它，要看 `DISABLE_AUTOUPDATER` 这类环境变量，而不是只盯着包管理器。

---

## Core Concepts

### How Claude Code works：agentic loop 是整个系统的骨架

Claude Code 是一个跑在终端里的 coding agent，它的每次交互都遵循同一套循环：读取请求 → 推理要做什么 → 调用 tool → 观察结果 → 重复，直到任务完成或需要用户输入。


**agentic loop 的完整流程**

**Step 1：用户发送消息**

你在终端里输入消息（交互模式），或者通过 `--print` / stdin 传入（非交互/headless 模式）。消息被追加进对话历史。

**Step 2：context 组装**

调用模型前，Claude Code 会先构建 system prompt，包括：当前日期、git 状态（分支、最近提交、工作区状态）、已加载的 CLAUDE.md memory 文件、以及可用 tool 列表。这个 context 在每次对话里只构建一次并做 memoize。

**Step 3：Claude 推理并选择 tool**

组装好的对话内容发给 Anthropic API。模型推理任务，并输出一个或多个 `tool_use` block，每个 block 指定 tool 名称和结构化 JSON 输入。

**Step 4：权限检查**

每个 tool 调用在真正执行前都会走 `checkPermissions`。根据当前 permission mode 和配置的 allow/deny 规则，结果是自动通过、弹出确认框，或者直接拒绝。

**Step 5：tool 执行并返回结果**

通过权限检查的 tool 运行。结果——文件内容、命令输出、搜索结果——以 `tool_result` block 的形式追加到对话。

**Step 6：循环继续**

模型收到 tool 结果后，要么继续调用更多 tool，要么输出最终文字回复。循环持续，直到某个模型 turn 里不再有 tool 调用为止。

> **关键点**：整个循环完全跑在你的本地终端进程里。没有什么远程执行 server 替你操作代码仓库。只有当 tool 显式发出网络请求时（如 `WebFetch`、`WebSearch`，或某个 MCP server），信息才会经过网络传出去。


**context 加载：system context 与 user context**

每次对话开始时，Claude Code 会组装两块 context，附加在每次 API 调用的前面。

**system context**（由 `getSystemContext()` 组装）：

- **Git 状态**：当前分支、默认/main 分支、git 用户名、`git status --short` 输出（超过 2000 字符会截断）、`git log --oneline` 最近 5 条提交
- **Cache-breaking injection**：内部调试用的临时注入字符串

设置了 `CLAUDE_CODE_REMOTE=1` 时，git 状态会被跳过。

**user context**（由 `getUserContext()` 组装）：

- **CLAUDE.md memory**：通过四层层级发现的所有 memory 文件（详见第二节）
- **当前日期**：以 `Today's date is YYYY-MM-DD` 的格式注入，让模型始终知道日期

两块 context 都用 `lodash/memoize` 在对话期间缓存。调用 `setSystemPromptInjection()` 会立即清空缓存。


**tool 执行模型**

Claude Code 默认不会自主执行 tool 调用。每个 tool 有一个 `checkPermissions` 方法，结果决定后续行为：

| 权限结果 | 行为 |
| --- | --- |
| `allow` | tool 立即运行，结果追加进对话 |
| `ask` | Claude Code 暂停，渲染确认弹窗 |
| `deny` | tool 调用被拒绝，Claude 收到一个错误结果 |

在 `bypassPermissions` 模式下，所有检查都跳过。`acceptEdits` 模式下，文件编辑 tool 自动通过，但 bash 命令仍然会弹确认。

只读、安全的 tool（如 `Read`、`Glob`、`Grep`）在所有模式下一般都会自动通过。


**交互模式 vs 非交互（task）模式**

**交互（REPL）模式**：默认体验。Claude Code 用 React/Ink 渲染实时终端 UI，你能看到流式输出、tool 使用确认和加载动画。消息在整个 session 里持续保留，直到你退出。

**非交互 / print 模式**：通过 `--print` 或 stdin 管道激活。不渲染 UI，输出直接写到 stdout，可被脚本或 CI pipeline 捕获。适合一次性自动化任务。

**Sub-agent（Task tool）**：Claude 可以通过 `Task` tool（`AgentTool`）启动 sub-agent。每个 sub-agent 跑自己独立的 agentic loop，有隔离的对话上下文，可选限制工具集。Sub-agent 可以在本地（in-process）或远程计算资源上运行。完成后，结果作为 tool result 返回给父 agent。


**对话存储与恢复**

对话以 JSON transcript 文件存在磁盘上（默认在 `~/.claude/`）。每个对话有唯一 session ID。用 `--resume <session-id>` 恢复指定对话，或直接 `--resume` 从列表选择。

恢复时会：
- 从磁盘加载完整消息历史
- 重新发现 memory 文件（可能与最初启动时不同）
- 将 permission mode 重置为配置的默认值（除非 session 里有持久化）

长对话会定期做 compaction：把最早的消息总结成摘要，以维持上下文窗口可用。磁盘上的完整原始 transcript 始终保留，compaction 只影响发给 API 的内容。

每个 tool 有 `maxResultSizeChars` 属性。结果超限时，内容会存入临时文件，模型收到预览加文件路径，防止大输出把上下文窗口撑爆。

---

### Memory and context（CLAUDE.md）：分层 memory 机制

Claude Code 的 memory 系统基于普通 Markdown 文件。在文件系统不同层级写 `CLAUDE.md`，就能在全局、项目、用户等不同范围定制 Claude 的行为——一些指令可以提交进代码库共享，另一些只在本地生效。


**四层 memory 层级**

Memory 文件按以下顺序加载（从低到高优先级）。越晚加载的文件优先级越高，因为模型对出现在上下文靠后位置的指令更敏感。

| 层级 | 路径 | 说明 |
| --- | --- | --- |
| **Managed memory**（最低优先级）| `/etc/claude-code/CLAUDE.md` | 管理员或部署工具统一设置的系统级指令，适用于机器上所有用户，支持 `rules/` 子目录，配合策略设置时不可被用户或项目文件覆盖 |
| **User memory** | `~/.claude/CLAUDE.md` 和 `~/.claude/rules/*.md` | 用户私有全局指令，适用于所有项目。放个人偏好，不会提交到任何仓库 |
| **Project memory** | `CLAUDE.md`、`.claude/CLAUDE.md`、`.claude/rules/*.md`（从根目录到 CWD 的每个祖先目录都检查）| 提交进代码库、全队共享的指令。放项目规范、架构说明、测试命令等 |
| **Local memory**（最高优先级）| `CLAUDE.local.md`（从根目录到 CWD 的每个祖先目录检查）| 私有的项目级覆盖，应加入 `.gitignore`。放本地环境路径、个人快捷方式等不想共享的偏好 |

离当前工作目录越近的文件，加载越晚，优先级越高。项目根目录的 `CLAUDE.md` 权重高于父目录的。


**文件发现算法**

Claude Code 启动时，从当前工作目录向上遍历到文件系统根，在每一层收集 memory 文件。发现顺序保证低优先级文件先出现在组装的 context 里：

1. Managed 文件最先加载
2. User 文件其次
3. Project 和 local 文件按从根目录到 CWD 的路径迭代——祖先目录先于子目录

完整文件列表在对话期间缓存（memoized）。用 `/memory` 打开 memory 编辑器可以强制重新加载，或者重启 session 来拾取外部修改。


**`@include` 指令**

Memory 文件可以用 `@` 符号引用其他文件。被引用的文件会作为独立条目插入到引用它的文件之前。

```markdown
# My project CLAUDE.md

@./docs/architecture.md
@./docs/conventions/typescript.md

Always run `bun test` before committing.
```

支持的路径形式：

| 语法 | 解析为 |
| --- | --- |
| `@filename` | 相对于引用文件所在目录 |
| `@./relative/path` | 显式相对路径 |
| `@~/home/path` | 相对于用户 home 目录 |
| `@/absolute/path` | 绝对路径 |

规则：
- 处于 fenced code block 和行内代码 span 内的 `@include` 路径会被忽略，只处理纯文本节点
- 检测到循环引用时会跳过
- 不存在的文件静默忽略
- 最大 include 深度为 5 层
- 只支持文本类文件（`.md`、`.ts`、`.py`、`.json` 等），图片和 PDF 等二进制文件会被跳过
- 默认情况下，引用项目目录之外的路径需要明确批准


**`.claude/rules/*.md`：细粒度规则文件**

不必把所有内容都塞进一个大 `CLAUDE.md`，可以把指令拆分到 `.claude/rules/` 下的多个 Markdown 文件：

```
my-project/
├── CLAUDE.md
└── .claude/
    └── rules/
        ├── testing.md
        ├── typescript-style.md
        └── git-workflow.md
```

`.claude/rules/` 下（含子目录）的所有 `.md` 文件会自动作为 project 和 user 层的 memory 条目加载。规则文件还支持 frontmatter 路径过滤：

```markdown
---
paths:
  - "src/api/**"
  - "src/services/**"
---

Always use dependency injection. Never import concrete implementations directly.
```

设置了 `paths` 后，只有在操作匹配 glob 模式的文件时，这条规则才会注入到 context 里。这在大型仓库里能有效控制 context 体积。


**单文件大小限制**

单个 memory 文件推荐不超过 40000 字符（`MAX_MEMORY_CHARACTER_COUNT`）。超限的文件会被标记，Claude 可能无法读取完整内容。保持 memory 文件聚焦、精简。


**Memory 如何影响 Claude 的行为**

Memory 文件加载后，会被组装成一个 context block，前面附上：

> "Codebase and user instructions are shown below. Be sure to adhere to these instructions. IMPORTANT: These instructions OVERRIDE any default behavior and you MUST follow them exactly as written."

这意味着 `CLAUDE.md` 文件里的指令会覆盖 Claude 的内置默认行为。


**各层级的典型用途**

- **Managed memory**：组织范围的策略、安全护栏、或特定部署的配置。由系统管理员管理。
- **User memory**（`~/.claude/CLAUDE.md`）：跨项目的个人偏好：响应语言、commit message 风格、个人别名等。这是私有的，永远不会提交。
- **Project memory**（`CLAUDE.md`）：与团队共享的项目规范：怎么跑测试、构建命令、架构决策、命名规范、PR checklist。提交这个文件。
- **Local memory**（`CLAUDE.local.md`）：特定项目的个人覆盖：本地环境路径、个人调试笔记、与团队默认不同的工作流偏好。加入 `.gitignore`。


**禁用 memory 加载**

| 方式 | 效果 |
| --- | --- |
| `CLAUDE_CODE_DISABLE_CLAUDE_MDS=1` | 完全禁用所有 memory 文件加载 |
| `--bare` flag | 跳过从 CWD 遍历自动发现 memory 文件；只加载 `--add-dir` 显式指定目录里的文件 |
| `claudeMdExcludes` 设置 | 要跳过的 memory 文件路径的 glob 模式（例如排除某个嘈杂的祖先目录 `CLAUDE.md`）|


**`/memory` 命令**

在 Claude Code REPL 里运行 `/memory` 可以打开 memory 文件编辑器。它会显示当前加载的 memory 文件，支持直接编辑，保存后重新加载 context。

也可以直接对 Claude 说："Add a rule to CLAUDE.md that we always use 2-space indentation."，Claude 会找到对应的 memory 文件并写入指令。

---

### Permissions：真正决定 agent 能不能放心用的底座

Claude Code 在你的本地机器上跑 tool——执行 shell 命令、编辑文件、抓取 URL。permission system 让你精确控制哪些操作 Claude 可以自动执行，哪些需要明确批准。


**权限管辖的操作类别**

- **文件操作**：通过 `Read`、`Edit`、`Write` tool 对本地文件系统进行读、编辑、写操作
- **Bash 命令**：通过 `Bash` tool 执行的任何 shell 命令，包括安装、构建、git 操作和任意脚本
- **MCP tool 调用**：连接的 MCP server 暴露的 tool，可能包括数据库查询、API 调用或浏览器自动化


**Permission mode 详解**

Permission mode 决定当没有具体 allow/deny 规则匹配某个 tool 调用时的默认行为。设置一次，整个 session 生效。

**`default`——对潜在危险操作询问**

标准模式。Claude Code 对每个 tool 调用进行评估，对可能有副作用的操作弹出确认——运行 shell 命令、编辑文件、发出网络请求。只读操作（文件读、搜索）自动通过。**推荐日常使用。**

**`acceptEdits`——自动批准文件编辑**

文件编辑和写操作（`Edit`、`Write`）无需确认自动通过。Bash 命令仍需确认。适合信任 Claude 自由改文件但还想审查 shell 命令的场景。

**`plan`——只读规划模式**

Claude 可以读文件、搜索代码库、讨论改动，但不能执行任何写入或 bash 操作，所有变更性 tool 调用都被阻止。想让 Claude 分析问题并出方案、再批准实际改动时用这个模式。模型可以通过 `ExitPlanMode` tool 请求退出 plan 模式。

**`bypassPermissions`——跳过所有权限检查**

所有权限检查禁用。每个 tool 调用立即执行，没有任何确认弹窗。**仅用于已提前审计过 Claude 行为的全自动脚本工作流。永远不要在交互 session 里用这个模式。**

**`dontAsk`——压制弹窗**

类似 `bypassPermissions`，但走稍微不同的内部路径。原本会弹确认的 tool 调用自动通过。适用于脚本/非交互场景。

**`auto`——transcript 分类器模式（feature-gated）**

实验性模式，用二级 AI 分类器对每个拟议的 tool 调用与对话 transcript 进行比对评估。分类器判断操作是否在请求范围之内，然后自动批准或交给人工确认。仅在启用了 `TRANSCRIPT_CLASSIFIER` feature flag 时可用。


**如何设置 permission mode**

**CLI flag**：启动时传入 `--permission-mode`：

```bash
claude --permission-mode acceptEdits
claude --permission-mode bypassPermissions
claude --permission-mode plan
```

**`/permissions` 命令**：在 session 中途无需重启即可切换模式，交互菜单选择，立即生效。

**`settings.json`**：在 user 或 project settings 里设置持久默认值：

```json
{
  "defaultMode": "acceptEdits"
}
```

有效值：`"default"`、`"acceptEdits"`、`"bypassPermissions"`、`"plan"`、`"dontAsk"`。


**细粒度 allow/deny 规则**

除全局 mode 外，还可以创建精细规则，对特定 tool 调用始终允许或始终拒绝——不论当前 mode 是什么。

每条规则有三个字段：

| 字段 | 说明 |
| --- | --- |
| `toolName` | 规则适用的 tool 名称（如 `"Bash"`、`"Edit"`、`"mcp__myserver"`）|
| `ruleContent` | 可选的模式，必须匹配 tool 的输入（如 Bash 的命令前缀）|
| `behavior` | `"allow"`、`"deny"` 或 `"ask"` |

规则在 permission mode 之前被评估。如果匹配到规则，直接应用其行为。

**规则来源与持久化**：

| 来源 | 存储位置 | 范围 |
| --- | --- | --- |
| `userSettings` | `~/.claude/settings.json` | 当前用户所有项目 |
| `projectSettings` | `.claude/settings.json` | 该项目的所有用户 |
| `localSettings` | `.claude/settings.local.json` | 当前用户，仅限该项目 |
| `session` | 内存 | 仅当前 session |
| `cliArg` | CLI flag | 仅当次调用 |

**示例：始终允许特定 git 命令**：

```json
{
  "permissions": {
    "allow": [
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Read(*)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Bash(sudo *)"
    ]
  }
}
```


**Bash 权限的工作方式**

Bash 命令权限检查值得单独说，因为 shell 命令可能很复杂。

**模式匹配**：`Bash` 规则的 `ruleContent` 字符串用通配符模式与命令匹配：

- `git status` — 仅精确匹配
- `git *` — 匹配任意 git 子命令
- `npm run *` — 匹配任意 `npm run` 脚本
- `*` — 匹配任意 bash 命令（谨慎使用）

**复合命令**：包含多个子命令（`&&`、`||`、`;`、`|`）的 bash 命令，每个子命令独立检查。整体权限取最严格的结果：任何一个子命令被拒绝，整条复合命令就被阻止。

**操作符限制**：某些 shell 结构无论规则如何都会受到额外审查：

- 指向项目目录之外路径的输出重定向（`>`、`>>`）
- 将当前目录 `cd` 到工作树之外
- `sed -i` 就地编辑命令（特殊处理以跟踪文件修改）

**安全检查**：无论当前 permission mode 如何，某些操作始终会被阻止或升级处理：

- 针对 `.claude/` 或 `.git/` 配置目录的命令
- 对 shell 配置文件（`.bashrc`、`.zshrc` 等）的修改
- 试图用跨平台路径技巧绕过路径限制的操作


**MCP tool 权限**

MCP tool 遵循和内置 tool 相同的规则系统。可以允许或拒绝整个 MCP server 或 server 内的单个 tool：

```json
{
  "permissions": {
    "deny": [
      "mcp__myserver"
    ],
    "allow": [
      "mcp__myserver__read_database"
    ]
  }
}
```

把 `mcp__servername`（不含具体 tool 名）作为规则，会整体拒绝该 server 的所有 tool——它们甚至不会出现在模型能看到的 tool 列表里。


**安全建议**

- 任何交互 session 都从 `default` 模式开始
- 探索陌生代码库或设计大改动时用 `plan` 模式——先看方案，再授权写入
- 信任 Claude 自由编辑文件但想审查 shell 命令时用 `acceptEdits`
- 优先用精细 allow 规则而非大幅提升 mode。允许 `Bash(git *)` 比切换到 `bypassPermissions` 安全得多
- `bypassPermissions` 和 `dontAsk` 只在隔离环境（容器、CI sandbox）里使用
- 克隆陌生仓库时检查 `.claude/settings.json`——里面可能预配置了权限规则

---

### Tools：Claude Code 给模型暴露了哪些"手脚"

Claude Code 给 Claude 提供了一套内置 tool，每个 tool 调用都受当前 permission mode 和规则约束。


**文件 tool**

**`Read`**

读取本地文件系统上的文件。默认读取最多 2000 行，支持 `offset` 和 `limit` 参数，以 `cat -n` 格式（带行号）返回内容。支持图片（PNG、JPG 等）视觉输入、PDF（每次最多 20 页）、Jupyter notebook（`.ipynb`）。

只读。在 `default` 模式下始终自动通过。

**`Edit`**

对文件做精确字符串替换。需要在同一对话里先调用 `Read`。用 `old_string` 替换为 `new_string`——匹配必须在文件里唯一命中。用 `replace_all: true` 可以在整个文件内重命名。

如果 `old_string` 出现不止一次（未设置 `replace_all`）则失败。这个约束大幅降低了 agent 改错位置的概率。

**`Write`**

创建新文件或完整覆盖已有文件。对已有文件，同样需要在同一对话里先调用 `Read`。修改已有文件时优先用 `Edit`——`Write` 会发送整个文件内容，更适合新建文件或完整重写。

**`Glob`**

按名称模式查找文件。快速模式匹配，适用于任意规模的代码库，返回的匹配文件路径按修改时间排序（最近修改的在前）。支持 `**/*.ts`、`src/**/*.test.js`、`**/CLAUDE.md` 等模式。

只读。始终自动通过。


**Shell tool**

**`Bash`**

在持久 shell session 里执行命令。Shell 在同一对话的多次 tool 调用之间保持状态——环境变量和工作目录变化会延续到后续调用。支持 `timeout` 参数。

关键行为：

- 复合命令（`&&`、`||`、`;`、`|`）会被解析，每个子命令独立进行权限检查
- 后台执行：传入 `run_in_background: true` 可运行长时任务而不阻塞，完成时会收到通知
- 输出限制：超出每工具结果大小预算的 stdout/stderr 会被截断，返回预览和文件路径
- 搜索命令（`find`、`grep`、`rg`）：内容搜索优先用专用的 `Grep` tool

在 `default` 模式下会弹权限确认。`acceptEdits` 模式下仅对有 allow 规则覆盖的命令自动通过。


**搜索 tool**

**`Grep`**

用正则表达式搜索文件内容，底层基于 ripgrep。支持完整正则语法（`log.*Error`、`function\s+\w+`），文件类型过滤（`*.ts`、`**/*.py`），三种输出模式：

- `files_with_matches`（默认）：只返回文件路径
- `content`：返回带上下文的匹配行
- `count`：返回每个文件的匹配数量

支持 `multiline: true` 进行多行模式匹配。只读，始终自动通过。

**`LS`**

列出目录内容，以结构化格式返回文件和子目录。在读取或编辑文件前探索项目结构时很有用。只读，始终自动通过。

在极简模式（`CLAUDE_CODE_SIMPLE=1`）下只有 `Bash`、`Read`、`Edit` 可用，用 `Bash` 执行 `ls` 代替。


**Web tool**

**`WebFetch`**

抓取指定 URL 并从中提取信息。接受 URL 和描述要提取什么的 prompt，将 HTML 转成 Markdown 后通过二级模型生成聚焦的回答。

特性：
- HTTP URL 自动升级为 HTTPS
- 15 分钟自清理缓存——重复抓取同一 URL 很快
- URL 重定向到不同 host 时返回重定向 URL，供后续请求使用
- GitHub URL 优先用 `gh` CLI 通过 `Bash`（如 `gh pr view`、`gh api`）

在 `default` 模式下弹权限确认。

**`WebSearch`**

搜索网络并返回结果。以 Markdown 链接格式返回标题、摘要和 URL。回答后 Claude 会自动附上引用了哪些 URL 的 `Sources:` 区块。支持域名过滤（包含或排除特定站点）。目前仅在美国可用。

在 `default` 模式下弹权限确认。


**Agent 和任务 tool**

**`Task`（Agent）**

启动一个 sub-agent 完成任务。在独立 context 里启动嵌套的 agentic loop，sub-agent 有自己的对话历史和工具集（可以受限），运行到完成后把结果返回给父 agent。

Sub-agent 可以：
- **本地运行**：in-process，共享父 agent 的文件系统和 shell
- **远程运行**：在满足条件时跑到独立计算资源上

适合开放式多步搜索、并行工作流，或把不同子问题委托给隔离的 agent。父 agent 收到的是 sub-agent 最终输出作为 tool result。

**`TodoWrite`**

创建和管理结构化任务列表。把带状态（`pending`、`in_progress`、`completed`）的 todo 条目写入终端 UI 里的持久面板，帮助 Claude 跟踪复杂多步任务的进度，也让你能看到它在做什么。

有 3 个以上独立步骤的任务建议主动使用。简单的单步请求不需要。结果渲染在 todo 面板里，不进入对话 transcript。


**MCP tool**

MCP（Model Context Protocol）server 可以给 Claude Code 暴露额外的 tool。连接的 tool 和内置 tool 并列出现，遵循同一套权限系统。

MCP tool 命名带 `mcp__` 前缀：

```
mcp__<server-name>__<tool-name>
```

例如，名为 `mydb` 的 server 上的 `query` tool 显示为 `mcp__mydb__query`。

常见的 MCP tool 类别：数据库查询和管理、浏览器和 web 自动化、云平台 API（AWS、GCP、Azure）、Issue tracker 集成（GitHub、Linear、Jira）、公司内部 tool 和 API。

MCP server 在 `~/.claude/mcp_servers.json` 或 `.claude/mcp_servers.json` 里配置。Server 运行并连接后，其 tool 自动加入 Claude 的上下文。

拒绝某个 MCP server 的所有 tool：

```json
{
  "permissions": {
    "deny": ["mcp__untrusted-server"]
  }
}
```


**Notebook tool**

**`NotebookEdit`**

编辑 Jupyter notebook 里的 cell，支持以行级精度插入、替换或删除 `.ipynb` 文件里的 cell。读取 notebook 用标准 `Read` tool，它会返回所有 cell 及其输出。


**Tool 可用性**

不是所有 tool 在所有 context 下都可用，活跃工具集在启动时确定：

- `CLAUDE_CODE_SIMPLE=1`：限制为只有 `Bash`、`Read`、`Edit`
- permission deny 规则：被规则整体拒绝的 tool 在模型看到 tool 列表前就被移除
- `isEnabled()` 检查：每个 tool 可以根据环境条件自我禁用（如 `WebSearch` 受地区限制）
- MCP server 连接状态：MCP tool 仅在 server 运行并已连接时可用

在 REPL 里用 `/tools` 命令查看当前活跃的工具集。

---

## Guides

### Authentication：三种接入方式

Claude Code 支持多种认证方式，取决于你是直接通过 Anthropic、通过云平台，还是通过 API key 辅助脚本来访问 API。


**Claude.ai OAuth（默认）**

首次运行 `claude` 且没有配置任何 API key 时，Claude Code 会用你的 claude.ai 账户启动 OAuth 流程。

1. 打开终端，运行 `claude`
2. Claude Code 显示一个 URL，提示你在浏览器里打开
3. 访问 URL，登录 claude.ai 账户，授权
4. 浏览器授权后，Claude Code 自动收到 OAuth token，存入安全存储（macOS 上是 Keychain，其他平台是 credentials 文件）

OAuth token 会在过期前自动续期，不需要重新认证（除非你主动登出或撤销授权）。


**API key 认证**

**环境变量**：在 shell profile 里或运行 `claude` 前设置 `ANTHROPIC_API_KEY`：

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

设置了这个变量后，Claude Code 直接使用它，不再触发 OAuth 流程。

**Settings 文件**：在 `~/.claude/settings.json` 里配置 `apiKeyHelper`，运行一个输出 API key 的 shell 命令。Claude Code 执行这个命令并缓存结果 5 分钟（可通过 `CLAUDE_CODE_API_KEY_HELPER_TTL_MS` 配置）：

```json
{
  "apiKeyHelper": "cat ~/.anthropic/api-key"
}
```

`apiKeyHelper` 命令必须只向 stdout 打印 API key，且以退出码 0 退出。任何 stderr 输出都被视为错误。

设置了 `ANTHROPIC_API_KEY` 或 `apiKeyHelper` 后，OAuth 流程会禁用，Claude Code 不再尝试使用 claude.ai 账户。


**AWS Bedrock**

通过 Amazon Bedrock 使用 Claude，设置 `CLAUDE_CODE_USE_BEDROCK` 环境变量并配置 AWS credentials：

```bash
export CLAUDE_CODE_USE_BEDROCK=1
```

Claude Code 使用标准 AWS credential chain，以下方式都支持：

- AWS credentials 文件（`~/.aws/credentials`）
- 环境变量：`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、`AWS_SESSION_TOKEN`
- IAM roles（EC2 instance profile、ECS task role 等）
- AWS SSO（`aws sso login`）

**自动刷新 AWS credentials**：如果 session 中途过期（如短效 SSO token），在 settings 里配置 `awsAuthRefresh`：

```json
{
  "awsAuthRefresh": "aws sso login --profile my-profile"
}
```

Claude Code 检测到 credentials 过期时会运行这个命令，并流式输出以便完成浏览器流程。

也可以用 `awsCredentialExport` 从命令导出 credentials：

```json
{
  "awsCredentialExport": "aws sts assume-role --role-arn arn:aws:iam::123456789012:role/MyRole --role-session-name claude-code --query Credentials --output json"
}
```

命令必须输出合法的 AWS STS JSON（含 `Credentials.AccessKeyId`、`Credentials.SecretAccessKey`、`Credentials.SessionToken`）。


**GCP Vertex AI**

通过 Google Cloud Vertex AI 使用 Claude，设置 `CLAUDE_CODE_USE_VERTEX` 并配置 Application Default Credentials：

```bash
export CLAUDE_CODE_USE_VERTEX=1
```

Claude Code 使用 Google Application Default Credentials（ADC），以下方式都支持：

- `gcloud auth application-default login`（交互使用）
- 通过 `GOOGLE_APPLICATION_CREDENTIALS` 指定 service account key 文件
- Workload Identity（GKE 上）

可选设置：

```bash
export ANTHROPIC_VERTEX_PROJECT_ID=my-gcp-project
export CLOUD_ML_REGION=us-central1
```

**自动刷新 GCP credentials**：配置 `gcpAuthRefresh`：

```json
{
  "gcpAuthRefresh": "gcloud auth application-default login"
}
```

Claude Code 在运行命令前会先检查当前 GCP credentials 是否有效，只有真正需要时才刷新。


**切换账户与登出**

- 运行 `/login` 启动新的 OAuth 流程，替换已存储的 token
- 运行 `/logout` 删除已存储的 credentials


**Token 过期与刷新**

OAuth token 自动处理续期，Claude Code 在每次 API 请求前检查 access token 是否过期，需要时自动刷新。多个并发 Claude Code 实例通过 lock 文件协调，避免重复刷新。如果 API 返回 `401`（如时钟偏差），Claude Code 会强制立即刷新。

刷新失败时（如在 claude.ai settings 里撤销了授权），Claude Code 会提示你运行 `/login`。


**认证优先级**

配置了多个认证来源时，按以下顺序解析（优先级从高到低）：

1. `ANTHROPIC_AUTH_TOKEN` 环境变量
2. `CLAUDE_CODE_OAUTH_TOKEN` 环境变量
3. 来自文件描述符的 OAuth token（托管部署用）
4. settings 里的 `apiKeyHelper`
5. 存储的 claude.ai OAuth token（keychain 或 credentials 文件）
6. `ANTHROPIC_API_KEY` 环境变量

CI 和非交互环境优先用 `ANTHROPIC_API_KEY` 或 `CLAUDE_CODE_OAUTH_TOKEN`，这两个在任何交互流程之前被检查。

---

### MCP servers：Claude Code 的外接总线

Model Context Protocol（MCP）是一个开放标准，让 Claude Code 连接外部数据源和服务。添加 MCP server 后，Claude 就能使用新的 tool——比如查询数据库、读取 Jira ticket、或与 Slack 工作区交互。


**添加 MCP server**

主要通过 `claude mcp add` CLI 命令，或直接编辑配置文件。

**通过 CLI**：

```bash
# 添加官方 filesystem MCP server
claude mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem /tmp

# 指定 scope
# 保存到当前目录的 .mcp.json（与团队共享）
claude mcp add --scope project filesystem -- npx -y @modelcontextprotocol/server-filesystem /tmp

# 保存到 user config（所有项目都可用）
claude mcp add --scope user my-db -- npx -y @my-org/mcp-server-postgres
```

**通过 `--mcp-config` flag**：启动 Claude Code 时传入 JSON 配置文件：

```bash
claude --mcp-config ./my-mcp-config.json
```

适合 CI 环境或需要自包含配置而不持久化到 settings 文件的场景。

> 这里只看 `mcp add` 很容易低估 MCP 的范围。对 Claude Code 来说，MCP 并不只是"远程工具调用协议"，它同时承担了 **tools、resources、prompts、认证** 四类扩展入口。


**配置文件格式**

MCP server 配置使用 JSON，顶层键为 `mcpServers`，每个 server 条目有名称和配置对象。

**Stdio（本地进程）**，通过 stdin/stdout 通信的本地子进程：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"],
      "env": {
        "NODE_ENV": "production"
      }
    }
  }
}
```

字段：`command`（必填）、`args`（命令行参数数组）、`env`（传给进程的环境变量）。

**HTTP（远程 server）**：

```json
{
  "mcpServers": {
    "my-api": {
      "type": "http",
      "url": "https://mcp.example.com/v1",
      "headers": {
        "Authorization": "Bearer $MY_API_TOKEN"
      }
    }
  }
}
```

字段：`type` 设为 `"http"`、`url`（必填）、`headers`（HTTP headers，值支持 `$VAR` 环境变量展开）。

**SSE（server-sent events）**：

```json
{
  "mcpServers": {
    "events-server": {
      "type": "sse",
      "url": "https://mcp.example.com/sse"
    }
  }
}
```

`command`、`args`、`url`、`headers` 中的值支持 `$VAR` 和 `${VAR}` 语法，在启动时从 shell 环境展开。变量缺失时 Claude Code 记录警告但仍尝试连接。


**配置 scope**

| Scope | 存储位置 | 用途 |
| --- | --- | --- |
| `project` | 当前目录（及往上的父目录）的 `.mcp.json` | 与团队共享的 server 配置 |
| `user` | `~/.claude.json`（全局 config）| 个人用，所有项目都可用 |
| `local` | 当前项目的 `.claude/settings.local.json` | 每项目的个人覆盖，不提交到版本控制 |

同名 server 出现在多个 scope 时，`local` > `project` > `user`。


**除了 tools，MCP 还会把 resources 与 prompts 带进来**

很多人第一次接触 MCP，只记住了"Claude 可以多几个 tool"。但在 Claude Code 里，MCP server 实际上还能暴露另外两类能力：

- **Resources**：可枚举、可读取的外部上下文，比如数据库 schema、工单详情、服务配置、内部文档片段
- **Prompts**：由 server 提供的参数化 prompt 模板，用来把某类任务包装成可复用的入口

这两类能力的意义在于，它们把"外部信息怎么进上下文"这件事结构化了。不是每次都靠 Claude 自己临时搜索或现编 prompt，而是让外部系统主动声明：这里有哪些上下文可读、有哪些工作流可调用。


**远程 MCP server 的认证与授权**

本地 stdio server 基本没有额外认证问题，但 HTTP / SSE server 常常需要鉴权。Claude Code 对这类 server 的处理并不是把认证硬编码在主程序里，而是把 server 状态显式分成：

- `connected`
- `pending`
- `failed`
- `needs-auth`
- `disabled`

其中 `needs-auth` 很关键。它意味着连接配置本身是对的，但 server 还没完成 OAuth 或其他授权流程。在这种状态下，Claude 看不到该 server 暴露出的 tool。

实际使用时，这一类问题通常不该靠猜，而应该先看：

```text
/mcp
```

如果 server 需要认证，再继续走授权流程，而不是误判成命令或网络问题。


**管理 server**

在 session 中途启用/禁用，无需重启或编辑配置文件：

```
/mcp enable <server-name>
/mcp disable <server-name>
/mcp enable all
/mcp disable all
```

强制重连：

```
/mcp reconnect <server-name>
```

查看所有已配置 server 及当前连接状态（`connected`、`pending`、`failed`、`needs-auth`、`disabled`）：

```
/mcp
```


**批准 MCP tool 调用**

Claude Code 在调用任何 MCP tool 前都会显示权限确认，展示 tool 名称和输入参数，供你在执行前审查。选项：

- **Allow once**：批准这次特定调用
- **Allow always**：本 session 内批准对该 tool 的所有调用
- **Deny**：阻止调用；Claude 收到错误后可以尝试其他方式

在 auto 模式（`--allowedTools`）下，可以通过在 allowed tools 列表里包含完整名称（`mcp__<server>__<tool>`）来预先批准 MCP tool。


**示例：filesystem server**

```bash
# 添加
claude mcp add --scope project filesystem -- npx -y @modelcontextprotocol/server-filesystem /home/user/projects

# 验证：运行 /mcp，确认 filesystem 显示为 connected
```

此后 Claude 可以用 `mcp__filesystem__read_file` 和 `mcp__filesystem__write_file` 读写 `/home/user/projects` 里的文件。


**示例：database server**

```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_CONNECTION_STRING": "$DATABASE_URL"
      }
    }
  }
}
```

启动 Claude Code 前在环境里设置 `DATABASE_URL`，MCP server 会自动收到。


**常见问题排查**

- **Server 显示为 'failed'**：检查命令是否存在（`which npx`），手动运行看是否有错误，确认必要的环境变量已设置，用 `claude --debug` 看详细连接日志
- **MCP tool 不出现**：connected 但未认证的 server 不会暴露 tool，检查 `/mcp` 里是否有 `needs-auth`，按 OAuth 流程授权
- **Windows 上 npx 启动失败**：需要 `cmd /c` 包装：`"command": "cmd", "args": ["/c", "npx", "-y", "@my-org/mcp-server"]`

---

### Hooks：事件驱动的自动化切口

Hooks 让你把自动化逻辑附加到 Claude Code 的 tool 生命周期上。Claude 读文件、跑 bash 命令、或结束回复时，你配置的 hook 会自动执行。用 hooks 来强制代码风格、跑测试、记录 tool 使用日志，或控制 Claude 允许做什么。


**Hook 的工作方式**

Hook 是一个命令（shell 脚本、HTTP 请求或 LLM prompt），绑定到特定事件上。事件触发时，Claude Code 运行所有匹配的 hook，用退出码和输出决定下一步。

每个 hook 的输入是 stdin 上的一个 JSON 对象，描述发生了什么——例如 `PreToolUse` 的 tool 名称和参数，或 `PostToolUse` 的 tool 名称和响应。

**退出码语义**（通用规律）：

| 退出码 | 含义 |
| --- | --- |
| `0` | 成功。stdout 可能展示给 Claude（取决于事件） |
| `2` | 阻止或注入。把 stderr 展示给 Claude，并（对 `PreToolUse`）阻止 tool 调用 |
| 其他 | 只把 stderr 展示给用户；执行继续 |

这里最容易漏掉的细节是：**stdout 并不总是回流给 Claude，是否回流取决于事件语义**。比如 `SessionStart`、`UserPromptSubmit`、`PreCompact` 这类事件，stdout 可以被当作新的上下文或指令；而很多纯观测型 hook，stdout 更像是给 transcript 或用户看的附加信息。这个设计避免了"任意 hook 输出都无条件污染主 prompt"。


**Hook 事件一览**

**`PreToolUse`**：每次 tool 调用前触发，输入包含 tool 名称和参数 JSON。退出 `2` 会阻止 tool 调用并把 stderr 展示给 Claude。用 matcher 限制只匹配特定 tool。

**`PostToolUse`**：每次成功 tool 调用后触发，输入包含 `inputs`（tool 参数）和 `response`（tool 结果）。退出 `0` 时 stdout 在 transcript 模式（Ctrl+O）里展示。退出 `2` 立即把 stderr 展示给 Claude。用来在文件编辑后跑 formatter、linter 或测试。

**`PostToolUseFailure`**：tool 调用出错时触发。输入包含 `tool_name`、`tool_input`、`error`、`error_type`、`is_interrupt`、`is_timeout`。退出码语义同 `PostToolUse`。

**`Stop`**：Claude 的 turn 结束前触发，不支持 matcher。退出 `2` 把 stderr 展示给 Claude 并继续对话（Claude 再得一个 turn）。可用来在 Claude 结束前检查所有必要任务是否完成。

**`SubagentStop`**：类似 `Stop`，但在 sub-agent（通过 Agent tool 启动的）结束时触发。输入包含 `agent_id`、`agent_type`、`agent_transcript_path`。

**`SubagentStart`**：新 sub-agent 启动时触发。输入包含 `agent_id`、`agent_type`。退出 `0` 时 stdout 展示给 sub-agent 的初始 prompt。

**`SessionStart`**：每个 session 开始时触发（启动、恢复、`/clear`、`/compact`），输入包含 `source`。退出 `0` 时 stdout 展示给 Claude。用 `source` 匹配：`startup`、`resume`、`clear`、`compact`。

**`UserPromptSubmit`**：你按 Enter 提交 prompt 时触发，输入包含你的原始 prompt 文本。退出 `0` 时 stdout 展示给 Claude（可以前置 context）。退出 `2` 阻止 prompt，只把 stderr 展示给用户。

**`PreCompact`**：对话 compaction 前触发（自动或手动）。退出 `0` 时 stdout 作为自定义 compact 指令追加。退出 `2` 阻止 compaction。用 `trigger` 匹配：`manual` 或 `auto`。

**`PostCompact`**：compaction 完成后触发，输入包含 compaction 详情和摘要。

**`Setup`**：用 `trigger: init`（项目初始化）或 `trigger: maintenance`（定期维护）触发。用于一次性设置脚本或定期维护任务。

**`PermissionRequest`**：权限确认弹窗出现时触发。输出含 `hookSpecificOutput.decision` 的 JSON 可以程序化地批准或拒绝。

**`PermissionDenied`**：auto 模式分类器拒绝 tool 调用后触发。返回 `{"hookSpecificOutput":{"hookEventName":"PermissionDenied","retry":true}}` 告诉 Claude 可以重试。

**`Notification`**：发送通知时触发（权限确认、空闲提示、认证成功、elicitation 事件）。用 `notification_type` 匹配。

**`CwdChanged`**：工作目录变更后触发，输入包含 `old_cwd` 和 `new_cwd`。环境变量 `CLAUDE_ENV_FILE` 已设置——向该文件写入 bash export 行可以把新环境变量应用到后续 Bash tool 调用。

**`FileChanged`**：监视的文件发生变化时触发，matcher 指定要监视的文件名模式（如 `.envrc|.env`）。同样支持 `CLAUDE_ENV_FILE`。

**`SessionEnd`**：session 结束时触发（clear、logout 或退出）。用 `reason` 匹配：`clear`、`logout`、`prompt_input_exit`、`other`。

**`ConfigChange`**：session 中 settings 文件变化时触发。用 `source` 匹配：`user_settings`、`project_settings`、`local_settings`、`policy_settings`、`skills`。退出 `2` 阻止变更被应用。

**`InstructionsLoaded`**：任何 instruction 文件（CLAUDE.md 或 rule）加载时触发。仅可观测，不支持阻止。

**`WorktreeCreate` / `WorktreeRemove`**：worktree 生命周期。`WorktreeCreate` 触发时 stdout 应包含创建的 worktree 的绝对路径。`WorktreeRemove` 在 worktree 需要清理时触发。

**`TaskCreated` / `TaskCompleted`**：任务创建或标记完成时触发。退出 `2` 阻止状态变更。

**`TeammateIdle`**：teammate 即将进入空闲状态前触发。退出 `2` 把 stderr 发给 teammate 并阻止其空闲。

**`Elicitation` / `ElicitationResult`**：MCP server 请求用户输入时触发。可程序化地提供响应。


**配置 hooks**

在 Claude Code 里运行 `/hooks` 打开交互式 hooks 配置菜单，或直接编辑 settings 文件：

- `~/.claude/settings.json`：user 级别（到处生效）
- `.claude/settings.json`：project 级别（该项目生效）
- `.claude/settings.local.json`：local hooks（不提交到 VCS）

**配置格式**：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "prettier --write $CLAUDE_FILE_PATH"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "echo 'Session complete' >> ~/.claude-log.txt"
          }
        ]
      }
    ]
  }
}
```

每个事件映射到一个 matcher 对象数组。每个 matcher 对象有：

- `matcher`（可选）：匹配事件 matchable 字段的字符串模式（tool 事件匹配 `tool_name`，`SessionStart` 匹配 `source`，`Setup` 匹配 `trigger`，`FileChanged` 匹配文件名模式）
- `hooks`：匹配时运行的 hook 命令数组


**Hook 命令类型**

**Shell command**：

```json
{
  "type": "command",
  "command": "npm test",
  "timeout": 60,
  "shell": "bash"
}
```

字段：`command`（必填）、`timeout`（秒）、`shell`（`"bash"` 或 `"powershell"`）、`statusMessage`（自定义加载文字）、`async`（后台运行不阻塞）、`once`（运行一次后自动移除）、`if`（permission rule 语法的条件跳过，如 `"Bash(git *)"`）。

**HTTP request**：

```json
{
  "type": "http",
  "url": "https://hooks.example.com/claude-event",
  "headers": {
    "Authorization": "Bearer $MY_TOKEN"
  },
  "allowedEnvVars": ["MY_TOKEN"],
  "timeout": 10
}
```

Claude Code 把 hook 输入 JSON POST 到该 URL。Headers 支持 `$VAR` 展开，变量须在 `allowedEnvVars` 里列出。

**LLM prompt**：

```json
{
  "type": "prompt",
  "prompt": "Review this tool call for security issues: $ARGUMENTS. If you find a problem, output an explanation and exit with code 2.",
  "model": "claude-haiku-4-5",
  "timeout": 30
}
```

Hook prompt 由 LLM 评估。`$ARGUMENTS` 替换为 hook 输入 JSON，LLM 的响应成为 hook 输出。

**Agent hook**：

```json
{
  "type": "agent",
  "prompt": "Verify that the unit tests in $ARGUMENTS passed and all assertions are meaningful.",
  "timeout": 60
}
```

类似 prompt hook，但作为完整 agent 运行，有 tool 访问权。适合需要读文件或运行命令的验证任务。


**实用示例**

**编辑后自动格式化**：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "prettier --write \"$CLAUDE_FILE_PATH\" 2>/dev/null || true"
          }
        ]
      }
    ]
  }
}
```

**bash 命令后跑测试**：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "if git diff --name-only HEAD | grep -q '\\.ts$'; then npm test; fi",
            "timeout": 120,
            "async": true
          }
        ]
      }
    ]
  }
}
```

**记录所有 tool 使用**：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "echo \"$(date -u +%Y-%m-%dT%H:%M:%SZ) $CLAUDE_TOOL_NAME\" >> ~/.claude-tool-log.txt",
            "async": true
          }
        ]
      }
    ]
  }
}
```

**阻止危险命令**：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "if echo \"$CLAUDE_TOOL_INPUT\" | grep -q 'rm -rf'; then echo 'Blocked: rm -rf is not allowed' >&2; exit 2; fi"
          }
        ]
      }
    ]
  }
}
```

**切换目录时注入环境变量**：

```json
{
  "hooks": {
    "CwdChanged": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "if [ -f .envrc ]; then grep '^export ' .envrc >> \"$CLAUDE_ENV_FILE\"; fi"
          }
        ]
      }
    ]
  }
}
```


**Hooks vs. Skills**

| 特性 | Hooks | Skills |
| --- | --- | --- |
| 何时运行 | 自动：tool 事件触发 | 显式：`/skill-name` 或 Claude 主动识别时调用 |
| 用途 | 副作用、控制、可观测性 | 按需工作流和能力 |
| 配置方式 | Settings JSON 里的 `hooks` 字段 | `.claude/skills/` 里的 Markdown 文件 |
| 输入 | 来自 tool 事件的 JSON | 你传给 skill 的参数 |

用 hooks 处理每次都要自动发生的事（格式化、日志记录、强制规范）；用 skills 处理你想按需触发的可复用工作流。

---

### Skills：把常用工作套路做成可复用能力包

Skills 是定义可复用 prompt 和工作流的 Markdown 文件。在 Claude Code 里输入 `/skill-name`，Claude 就会加载该 skill 的指令并执行描述的任务。Skills 适合所有跨 session 重复的工作流——跑一次部署、写 changelog、review PR、或套用团队特定的编码规范。


**Skills 的工作方式**

Skill 是 `.claude/skills/` 里含有 `SKILL.md` 文件的目录。当你调用 `/skill-name` 时，Claude Code 把该 skill 的 `SKILL.md` 作为操作的 prompt 加载。Skills 是懒加载的——只在调用时才读取，定义很多 skill 不影响启动时间和上下文大小。


**创建 skill**

```bash
# 创建 skill 目录
mkdir -p .claude/skills/my-skill
```

Skills 可以放在：
- `.claude/skills/`（project 级别，相对于当前工作目录）
- `~/.claude/skills/`（user 级别，所有项目都可用）

**编写 `SKILL.md`**（`.claude/skills/my-skill/SKILL.md`）：

```markdown
---
description: Run the full release process for this project
argument-hint: version number (e.g. 1.2.3)
---

Release the project at version $ARGUMENTS.

Steps:
1. Update the version in `package.json` to $ARGUMENTS
2. Update CHANGELOG.md with a new section for this version
3. Run `npm test` and confirm all tests pass
4. Commit with message "chore: release v$ARGUMENTS"
5. Create a git tag `v$ARGUMENTS`
```

调用：

```
/my-skill 1.2.3
```

Claude 加载 skill 并执行指令，`1.2.3` 替换 `$ARGUMENTS`。


**Skill frontmatter**

SKILL.md 顶部的 frontmatter 配置 skill 的行为，所有字段都是可选的：

```yaml
---
description: A short description shown in /skills and used by Claude to decide when to use it
argument-hint: what to pass as the argument (shown in autocomplete)
allowed-tools: Bash, Write, Read
when_to_use: Use this skill when the user asks to create a new component
model: claude-sonnet-4-6
user-invocable: true
context: fork
---
```

| 字段 | 说明 |
| --- | --- |
| `description` | 在 `/skills` 里展示的短描述，也供 Claude 判断何时调用该 skill |
| `argument-hint` | slash 命令自动补全里展示的参数提示 |
| `allowed-tools` | 该 skill 允许使用的 tool 列表（逗号分隔，默认全部）|
| `when_to_use` | 描述 Claude 应该主动使用该 skill 的场景 |
| `model` | 该 skill 使用的模型（如 `claude-sonnet-4-6`），默认用 session 模型 |
| `user-invocable` | 设为 `false` 则从 slash 命令列表里隐藏（Claude 仍可用）|
| `context` | `fork` 在隔离的 subagent context 里运行该 skill |
| `paths` | Glob 模式；只在操作匹配文件时激活该 skill |
| `version` | Skill 版本字符串 |
| `hooks` | 限定在该 skill 执行范围内的 hooks（格式同 settings hooks）|


**参数替换**

在 `SKILL.md` 任意位置用 `$ARGUMENTS` 插入 slash 命令后传入的文本：

```markdown
Create a new React component named $ARGUMENTS following the project's conventions.
```

```
/new-component UserProfile
```

对于命名参数，在 frontmatter 里列出参数，然后用 `$name` 的形式引用：

```yaml
---
arguments: [name, directory]
---
```


**行内 shell 命令**

Skills 可以用反引号注入语法嵌入在调用时执行的 shell 命令，输出在 Claude 看到之前就插入到 prompt 里：

```markdown
---
description: Review recent changes
---

Here are the recent commits for context:

!`git log --oneline -20`

Review the changes above and summarize what was accomplished.
```

`!` 前缀加反引号引用的命令会运行，并把输出替换该块。这对把实时项目状态注入 skill prompt 很有用。

行内 shell 命令在你调用 skill 时执行，不在 skill 加载时执行。


**列出 skills**

```
/skills
```

展示所有可用 skill（来自所有 scope：project、user、managed）及其描述。


**命名空间 skills**

子目录里的 skill 用冒号命名空间：

```
.claude/skills/
  deployment/
    SKILL.md      → /deployment
  database/
    migrate/
      SKILL.md    → /database:migrate
    seed/
      SKILL.md    → /database:seed
```


**路径条件激活（conditional skills）**

在 frontmatter 里加 `paths` 字段，只在操作匹配文件时激活 skill：

```yaml
---
description: Django model review
paths: "**/*.py"
when_to_use: Use when editing Django model files
---
```

操作匹配 glob 模式的文件时，skill 自动加载进 Claude 的 context，否则不占用 context。


**Bundled skills**

Claude Code 自带内置 skills，编译进二进制，启动时自动注册，在 `/skills` 里显示为 `bundled`。包括项目 onboarding 辅助、常见代码 review 工作流、agentic 搜索和分析模式等。


**User 级别 skills**

`~/.claude/skills/` 里的 skill 在所有项目里都可用，无需逐仓库添加，适合跨项目的个人工作流：

```bash
mkdir -p ~/.claude/skills/standup
cat > ~/.claude/skills/standup/SKILL.md << 'EOF'
---
description: Summarize what I worked on today for a standup update
---

Look at my git commits from today across this repository and summarize them in standup format: what I did, what I'm doing next, and any blockers. Keep it to 3-4 sentences.
EOF
```


**示例：自定义组件生成器**

```markdown
---
description: Generate a new React component with tests
argument-hint: ComponentName
allowed-tools: Write, Bash
---

Create a new React component named $ARGUMENTS.

1. Create `src/components/$ARGUMENTS/$ARGUMENTS.tsx` with:
   - A functional component using TypeScript
   - Props interface named `$ARGUMENTSProps`
   - JSDoc comment describing the component
   - Default export

2. Create `src/components/$ARGUMENTS/$ARGUMENTS.test.tsx` with:
   - At least one rendering test using React Testing Library
   - A snapshot test

3. Create `src/components/$ARGUMENTS/index.ts` that re-exports the component.

4. Run `npx tsc --noEmit` to confirm no type errors.
```

调用：

```
/new-component Button
```

---

### Multi-agent workflows：并行与隔离，而不是炫技

Claude Code 可以启动 sub-agent——独立运行以并行完成任务的独立 Claude 实例。这让你可以把大型、多步骤工作分发给专门的 agent 并发运行，而不是在单一对话里顺序完成所有事情。


**Sub-agent 的工作方式**

Claude 使用 `Agent` tool 时，会启动一个新的 Claude 实例，有自己的 context、system prompt 和工具权限。父 Claude 等待 agent 完成（或如果 agent 在后台运行，则继续其他工作），然后把 agent 的结果作为单条消息接收。

每个 sub-agent：
- 从全新的 context 窗口开始（除非是 fork）
- 根据 agent 类型获得专门的 system prompt
- 有自己的工具权限（可按 agent 类型配置）
- 本身可以继续启动 sub-agent（但嵌套有限制）


**Agent tool**

Claude 用 `Agent` tool 启动 sub-agent，你不直接调用这个 tool——Claude 自己决定何时使用。Claude 启动 agent 时，终端里会显示带独立进度指示器的 agent 条目。

Tool 参数：

- `description`：3-5 个词的任务摘要（在 UI 里显示）
- `prompt`：给 agent 的完整任务描述
- `subagent_type`：使用哪种专用 agent 类型（可选，默认通用）
- `run_in_background`：是否异步运行，让 Claude 可以继续其他工作
- `isolation`：`"worktree"` 给 agent 一个隔离的 git worktree


**Claude 何时使用 sub-agent**

Claude 在任务能从并行或专业化中受益时启动 agent：

- **独立并行任务**：在写测试的同时更新文档
- **专业化工作**：用代码 review agent 做安全审计
- **长时任务**：Claude 处理其他事情时在后台做研究
- **隔离探索**：fork 自己去探索方案而不污染主 context

对于简单任务、小文件读取、或几个 tool 调用能直接完成的事，Claude 不会启动 agent。


**请求 multi-agent 工作流**

你可以明确要求 Claude 使用多个 agent：

```
Run the linter and the test suite in parallel.
```

```
Use separate agents to research how three competing libraries handle this problem,
then summarize the findings.
```

```
Have an agent review the security implications of this code change while you
continue implementing the feature.
```

要求 Claude"并行"运行 agent 时，它会发一条含多个 Agent tool 调用的消息，同时启动所有 agent。要明确说清楚哪些应该并行、哪些必须顺序。


**前台 vs 后台 agent**

**前台 agent**（默认）：Claude 等待每个 agent 完成后再继续。当 Claude 需要结果才能继续工作时用前台 agent。

**后台 agent**：异步运行。Claude 启动后继续其他工作，agent 完成时收到通知：

```
Run the integration tests in the background while you implement the next feature.
```

Claude 不会主动轮询或检查后台 agent 的输出文件——它继续工作，agent 完成时自动收到结果通知。

后台 agent 显示在任务面板里，你可以看到进度并在需要时取消。


**Coordinator 模式**

Coordinator 模式下，Claude 充当 orchestrator，把所有实现工作委托给 sub-agent，自己专注于规划、路由和综合。适合超大型任务，希望 Claude 管理整体工作流、sub-agent 做具体工作的场景。

Coordinator 模式主要用于 multi-agent 团队设置。在标准单用户 session 里，Claude 自己决定何时委托。


**Agent memory 与 context 隔离**

每个 sub-agent 从全新的 context 窗口开始。父 Claude 需要在 agent prompt 里提供完整的任务描述和任何相关背景——agent 不会自动继承父对话历史。

这意味着：
- Agent 是独立的，不能读取父对话
- 父必须在 prompt 里提供充足的 context，让 agent 能成功完成任务
- Agent 的结果作为单条响应返回，不是逐轮流式的

**持久 agent memory**：某些 agent 类型跨调用有持久 memory，存储在 Markdown 文件里：

- User scope：`~/.claude/agent-memory/<type>/MEMORY.md`——跨所有项目共享
- Project scope：`.claude/agent-memory/<type>/MEMORY.md`——与团队共享
- Local scope：`.claude/agent-memory-local/<type>/MEMORY.md`——本地，不提交


**Worktree 隔离**

设置 `isolation: "worktree"` 给 agent 一个独立的 git worktree——仓库的隔离副本，agent 做的改动不影响你的工作目录，直到你合并：

```
Implement this feature in an isolated worktree so I can review the changes before merging.
```

agent 做了改动时，结果里会包含 worktree 路径和分支名，供你检查。没有改动时，worktree 自动清理。


**如何写出有效的 agent prompt**

Sub-agent 从父对话得到零 context。Claude 应该——你也可以提示 Claude——写出自包含的任务说明。

一个好的 agent prompt 包含：
- agent 要完成什么、为什么
- 相关文件路径、函数名或数据
- agent 应该汇报什么（格式、长度、要回答的具体问题）
- agent 不应该做什么（范围限制）
- 已经尝试或排除了什么

**弱 prompt**：

```
Based on your findings, fix the bug.
```

**强 prompt**：

```
Fix the null reference bug in UserService.getProfile() in src/services/user.ts:247.
The bug occurs when the user has no associated profile record — getProfile() calls
profile.preferences without checking if profile is null first. Add a null check and
return a default preferences object { theme: 'light', notifications: true }.
Run npm test afterward to confirm the fix passes existing tests.
```


**最佳实践**

- **给 agent 分配独立、可并行的任务**：agent 在工作不互相依赖时最有价值。如果 B 依赖 A 的结果，它们必须顺序——但 A 和 C 可以并行，B 等 A
- **在 prompt 里提供完整 context**：直接粘贴相关代码片段、文件路径、函数签名或错误信息。不要假设 agent 能从最少的指令推断出所需的一切
- **明确说清楚预期输出格式**：需要简短摘要就说"不超过 200 字"，需要结构化输出就描述格式
- **有风险的改动用 worktree 隔离**：agent 要做大量文件改动时，用 `isolation: "worktree"` 以便合并前审查
- **不要过度并行**：为小任务启动很多 agent 只会增加开销。信任 Claude 的判断，除非你脑子里有特定的并行结构


**限制与安全**

- Sub-agent 有自己的 permission mode，默认用 `acceptEdits` 模式
- 可以用 `Agent(AgentName)` 语法在 deny 规则里拒绝特定 agent
- 后台 agent 不与父的 abort controller 关联——按 Escape 取消父的 turn，但不会取消正在运行的后台 agent，用任务面板显式取消
- Sub-agent 不能启动其他 teammate（团队名单是扁平的），但可以用 Agent tool 继续启动 sub-agent
- Fork agent（继承父 context 的 agent）不能再 fork——Claude 阻止递归 forking
- Agent 结果返回给父之前上限为 100000 字符

---

## Configuration

### Settings：分层配置体系

Claude Code 从多个 scope 的 JSON 文件里读取 settings，从低到高优先级合并，更具体的 scope 覆盖更宽泛的。


**Settings 文件位置**

**全局（user）**：`~/.claude/settings.json`

适用于你跑的所有 Claude Code session、跨所有项目。在这里设置个人偏好：首选模型、主题、清理策略等：

```json
{
  "model": "claude-opus-4-5",
  "cleanupPeriodDays": 30,
  "permissions": {
    "defaultMode": "acceptEdits"
  }
}
```

**Project（共享）**：项目根目录的 `.claude/settings.json`

提交到版本控制。用于应该适用于所有项目贡献者的设置——permission 规则、hook 配置、MCP server、环境变量：

```json
{
  "permissions": {
    "allow": ["Bash(npm run *)", "Bash(git *)"],
    "deny": ["Bash(rm -rf *)"]
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [{ "type": "command", "command": "npm run lint" }]
      }
    ]
  }
}
```

**Local（个人项目）**：项目根目录的 `.claude/settings.local.json`

不提交到版本控制（自动加入 `.gitignore`）。用于特定项目内的个人覆盖——自己的 permission 偏好或不应共享的环境变量：

```json
{
  "permissions": {
    "defaultMode": "bypassPermissions"
  }
}
```

**Managed（企业）**：平台特定的系统路径

由管理员通过 MDM、注册表（Windows）、plist（macOS）或 managed settings 文件设置。Managed settings 优先级最高，用户或项目无法覆盖。


**打开 settings**

在任意 Claude Code session 里运行 `/config` 打开 settings UI。也可以直接编辑 JSON 文件，Claude Code 检测到文件变化时会自动重新加载。


**Settings 优先级**

从低到高合并，后来的 source 对标量值进行覆盖，数组合并并去重：

```
Plugin defaults → User settings → Project settings → Local settings → Managed (policy) settings
```

Managed（policy）settings 始终有最终优先权。

如果把 CLI flag 和运行时输入也算进来，更准确的心智模型应该是：

```text
内置默认值
→ 插件默认值
→ User settings
→ Project settings
→ Local settings
→ Managed settings
→ 启动 flag / 当前 session 的显式选择
```

也就是说，settings 决定的是"长期默认值"，而不是所有时候的最终值。像 `--model`、`--permission-mode` 这种启动参数，本质上属于更晚发生的运行时覆盖。


**Settings 参考**

**`model`** `string`：覆盖 Claude Code 使用的默认模型，接受配置的 provider 支持的任意模型 ID。

**`permissions`** `object`：控制 Claude 可以使用哪些 tool 以及用什么模式。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `allow` | `string[]` | Claude 可以不询问就执行的操作规则 |
| `deny` | `string[]` | Claude 始终被阻止执行的操作规则 |
| `ask` | `string[]` | 始终弹确认的操作规则 |
| `defaultMode` | `string` | 默认 permission mode |
| `disableBypassPermissionsMode` | `"disable"` | 阻止用户进入 bypass permissions 模式 |
| `additionalDirectories` | `string[]` | Claude 可以访问的额外目录 |

**`hooks`** `object`：在 tool 执行前后运行自定义 shell 命令。支持事件：`PreToolUse`、`PostToolUse`、`Notification`、`UserPromptSubmit`、`SessionStart`、`SessionEnd`、`Stop`、`SubagentStop`、`PreCompact`、`PostCompact`。

**`cleanupPeriodDays`** `integer`（默认 30）：保留 chat transcript 的天数。设为 `0` 完全禁用 session 持久化——不写入 transcript，启动时删除已有的。

**`env`** `object`：注入每个 Claude Code session 的环境变量：

```json
{
  "env": {
    "NODE_ENV": "development",
    "MY_API_URL": "https://api.example.com"
  }
}
```

**`availableModels`** `string[]`（仅 Managed）：企业用户可选模型白名单。接受系列别名（`"opus"` 允许任意 Opus 版本）、版本前缀或完整模型 ID。`undefined` 则所有模型可用，空数组则只有默认模型可用。

**`allowedMcpServers` / `deniedMcpServers`** `object[]`：MCP server 的企业白名单和黑名单，denylist 优先。

**`worktree`** `object`：`--worktree` flag 行为配置：

| 字段 | 说明 |
| --- | --- |
| `symlinkDirectories` | 从主仓库 symlink 到 worktree 的目录（如 `"node_modules"`）|
| `sparsePaths` | 用于大型 monorepo 加速 worktree 的 sparse checkout 路径 |

**`attribution`** `object`：自定义 Claude 附加到 commit 和 PR 描述上的署名文字：

```json
{
  "attribution": {
    "commit": "Co-Authored-By: Claude <noreply@anthropic.com>",
    "pr": ""
  }
}
```

**`language`** `string`：Claude 响应和语音听写的首选语言。

**`alwaysThinkingEnabled`** `boolean`（默认 `true`）：设为 `false` 禁用 extended thinking。

**`effortLevel`** `"low" | "medium" | "high"`：支持可调 thinking budget 的模型的持久 effort 级别。

**`autoMemoryEnabled`** `boolean`（User、local）：禁用时 Claude 不读写 auto-memory 目录。

**`claudeMdExcludes`** `string[]`：要排除不加载的 `CLAUDE.md` 文件的 glob 模式（用 picomatch 匹配绝对路径）：

```json
{
  "claudeMdExcludes": [
    "/home/user/monorepo/vendor/CLAUDE.md",
    "**/third-party/.claude/rules/**"
  ]
}
```

**`disableAllHooks`** `boolean`：设为 `true` 禁用所有 hook 和 `statusLine` 执行。

**`respectGitignore`** `boolean`（默认 `true`）：文件选择器是否遵守 `.gitignore` 文件。

**`syntaxHighlightingDisabled`** `boolean`：禁用 diff 里的语法高亮。

**`prefersReducedMotion`** `boolean`：减少或禁用动画（spinner 闪烁效果等）。

**`statusLine`** `object`：自定义底部状态栏展示内容，常用于把当前分支、测试状态、工作目录或外部脚本输出持续显示在 Claude Code UI 里。

**`outputStyle`** `string`：控制输出呈现风格，常见场景是脚本化运行、简化终端渲染，或配合低噪声的团队默认配置。

**`forceLoginMethod`** `string`：在团队或企业环境里强制要求某一种登录方式，比如只允许 API key、Bedrock、Vertex 或组织指定的认证入口。

**`enabledMcpjsonServers` / `disabledMcpjsonServers`**：控制从 `.mcp.json` 自动发现到的 server 中哪些默认启用、哪些默认禁用。它解决的是"项目里共享了很多 MCP server，但每个人不一定都想一上来全部连上"的问题。

**`sandbox`** `object`：集中配置 Claude Code 运行 shell/tool 时的沙箱策略。和 permission rules 不同，permission 决定"要不要让 Claude 试着做这件事"，sandbox 决定"即便允许了，它能在多大边界内做"。


**Managed settings（企业）**

管理员可以通过平台原生机制向所有用户推送 settings，Managed settings 高于所有用户和项目 settings。

- **macOS**：通过 MDM（Jamf、Kandji 等）部署 plist 文件到 `/Library/Preferences/`，目标为 `com.anthropic.claudecode`
- **Windows**：写入 `HKLM\Software\Anthropic\Claude Code` 注册表键（机器范围）或 `HKCU\Software\Anthropic\Claude Code`（用户级）
- **文件**：在平台特定的 managed 路径放置 `managed-settings.json`，也支持 `managed-settings.d/` 目录做 drop-in 片段：

```
managed-settings.json          # 基础设置（最低优先级）
managed-settings.d/
  10-security.json             # 最先合并
  20-model-policy.json         # 后合并（覆盖 10-）
```

仅 managed 可用的锁定设置：

| 设置 | 说明 |
| --- | --- |
| `allowManagedHooksOnly` | `true` 时只运行 managed settings 里定义的 hook |
| `allowManagedPermissionRulesOnly` | `true` 时只遵守 managed settings 里的 permission 规则 |
| `allowManagedMcpServersOnly` | `true` 时 `allowedMcpServers` 只从 managed settings 读取 |
| `strictPluginOnlyCustomization` | 把特定定制面（`"skills"`、`"agents"`、`"hooks"`、`"mcp"`）锁定为 plugin-only；传 `true` 锁定全部四项，或传数组指定 |


**JSON Schema**

在 settings 文件里加 `$schema` 启用编辑器自动补全和验证：

```json
{
  "$schema": "https://schemas.anthropic.com/claude-code/settings.json"
}
```

---

### CLAUDE.md 怎么写：决定 agent 像不像成熟同事

`CLAUDE.md` 文件让你把项目知识编码进去，Claude 在每个 session 开始时自动加载。不用每次都解释你的项目规范、构建系统和架构——写一次，Claude 自动读取。


**什么该写进 CLAUDE.md**

**写进去的**（缺了会导致错误的信息）：

- 构建、测试、lint 命令（具体调用方式，不只是工具名）
- 影响代码编写或组织方式的架构决策
- 项目特定的编码规范（命名模式、文件结构规则）
- 环境设置要求（必需的环境变量、预期的服务）
- Claude 应该知道要避免的常见坑或模式
- Monorepo 结构和各包负责什么

**不应该写的**：

- Claude 已知的内容（标准 TypeScript 语法、常见库 API）
- 显而易见的提示（"写干净代码"、"加注释"）
- 敏感数据——API key、密码、token 或任何 secret
- 频繁变化会变过时的信息

**判断标准**：删掉这行 Claude 会在你的代码库里犯错吗？不会的话就删掉。


**文件位置与层级**

| 文件 | 类型 | 用途 |
| --- | --- | --- |
| `/etc/claude-code/CLAUDE.md` | Managed | 管理员设置的系统级指令，适用于所有用户 |
| `~/.claude/CLAUDE.md` | User | 你的个人全局指令，适用于所有项目 |
| `~/.claude/rules/*.md` | User | 模块化全局规则，每个文件单独加载 |
| `CLAUDE.md`（项目根） | Project | 提交进版本控制的共享团队指令 |
| `.claude/CLAUDE.md`（项目根） | Project | 共享项目指令的替代位置 |
| `.claude/rules/*.md`（项目根） | Project | 按主题组织的模块化项目规则 |
| `CLAUDE.local.md`（项目根） | Local | 你的私有项目特定指令，不提交 |

`.claude/rules/` 支持子目录，递归找到的所有 `.md` 文件都会作为独立 memory 条目加载。


**加载顺序与优先级**

从低到高加载，越晚加载的文件有效优先级越高（模型对出现在 context 靠后位置的内容更敏感）：

1. **Managed memory**：`/etc/claude-code/CLAUDE.md` 和 `/etc/claude-code/rules/*.md`，始终加载，不可排除
2. **User memory**：`~/.claude/CLAUDE.md` 和 `~/.claude/rules/*.md`
3. **Project memory（从根到 CWD）**：从文件系统根到当前目录的每一层的 `CLAUDE.md`、`.claude/CLAUDE.md`、`.claude/rules/*.md`，离 CWD 越近的目录越后加载（优先级越高）
4. **Local memory**：从根到 CWD 的每一层的 `CLAUDE.local.md`，同样的遍历顺序，默认 gitignore


**Frontmatter 路径过滤**

`.claude/rules/` 里的文件可以用 YAML frontmatter 限制应用的文件路径，根据 Claude 正在操作的文件按条件加载规则：

```markdown
---
paths:
  - "src/api/**"
  - "*.graphql"
---

# API conventions

All API handlers must validate input using the shared `validate()` helper.
GraphQL resolvers must not perform direct database queries — use the data layer.
```

没有 frontmatter 的规则无条件应用，有 `paths` frontmatter 的只在 Claude 操作匹配 glob 模式的文件时应用。


**TypeScript 项目的 CLAUDE.md 示例**

```markdown
# Project: Payments API

## Build and test

- Build: `bun run build`
- Tests: `bun test` (uses Bun's built-in test runner — do not use Jest)
- Lint: `bun run lint` (biome, not eslint)
- Type check: `bun run typecheck`

Always run `bun run typecheck` before considering a change complete.

## Architecture

- `src/handlers/` — HTTP handlers, one file per route group
- `src/services/` — Business logic, no direct DB access
- `src/db/` — Database layer (Drizzle ORM); all queries live here
- `src/schemas/` — Zod schemas shared between handler validation and DB types

Handlers call services. Services call the DB layer. Never skip layers.

## Conventions

- Use `z.object().strict()` for all input validation schemas
- Errors propagate as `Result<T, AppError>` — never throw in service code
- All monetary values are integers in cents
- Timestamps are Unix seconds (number), not Date objects

## Environment

Required env vars: `DATABASE_URL`, `STRIPE_SECRET_KEY`, `JWT_SECRET`
Local dev: copy `.env.example` to `.env.local` and fill in values

## Common mistakes to avoid

- Do not use `new Date()` directly — use `getCurrentTimestamp()` from `src/utils/time.ts`
- Do not add `console.log` — use the `logger` from `src/utils/logger.ts`
- Do not write raw SQL — use the Drizzle query builder
```


**用 `/init` 生成初始 CLAUDE.md**

在任意 Claude Code session 里运行 `/init`，Claude 会分析你的代码库并生成包含最相关命令和 context 的 `CLAUDE.md`：

```
/init
```

生成的文件只是起点。审查它，删掉不真正有用的内容，加入 Claude 无法从代码推断出来的项目特定知识。


**排除文件**

有 `CLAUDE.md` 文件不想让 Claude 加载（如 vendor 依赖或生成的代码里的），在 settings 里加排除模式：

```json
{
  "claudeMdExcludes": [
    "/absolute/path/to/vendor/CLAUDE.md",
    "**/generated/**",
    "**/third-party/.claude/rules/**"
  ]
}
```

模式用 picomatch 匹配绝对路径，只有 User、Project、Local memory 类型可以排除，Managed（管理员）文件始终加载。

---

### Environment variables：脚本化、CI 化、远程化的总控制面

Claude Code 在启动时读取环境变量，用于配置认证、指向自定义 API endpoint、调整运行时行为，以及控制哪些功能处于活跃状态——无需修改 settings 文件。


**认证**

| 变量 | 说明 |
| --- | --- |
| `ANTHROPIC_API_KEY` | 直接与 Anthropic API 认证的 API key。设置后使用此 key，不触发 OAuth |
| `ANTHROPIC_BASE_URL` | 覆盖 Anthropic API 的 base URL，用于指向代理、staging 环境或兼容的第三方 endpoint |
| `ANTHROPIC_AUTH_TOKEN` | 自定义 `Authorization` header 的 bearer token，常用于非标准代理或中间层鉴权 |
| `ANTHROPIC_CUSTOM_HEADERS` | 以文本形式附加自定义请求头，适合经过企业网关或审计代理时使用 |
| `CLAUDE_CODE_USE_BEDROCK` | 设为 `1` 或 `true`，使用 AWS Bedrock 作为 API provider |
| `CLAUDE_CODE_USE_VERTEX` | 设为 `1` 或 `true`，使用 Google Vertex AI 作为 API provider |
| `CLAUDE_CODE_OAUTH_TOKEN` | 直接用作认证的 OAuth access token，绕过交互式登录流程，适合自动化环境 |

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export ANTHROPIC_BASE_URL="https://my-proxy.example.com"
```


**配置路径**

| 变量 | 说明 |
| --- | --- |
| `CLAUDE_CONFIG_DIR` | 覆盖 Claude Code 存储配置、settings 和 transcript 的目录，默认 `~/.claude` |
| （managed settings 路径变量）| 覆盖 managed settings 文件路径，适用于默认平台路径不合适的企业环境 |

```bash
export CLAUDE_CONFIG_DIR="/opt/claude-config"
```


**模型选择**

| 变量 | 说明 |
| --- | --- |
| `ANTHROPIC_MODEL` | 使用的默认模型，会被 settings 文件里的 `model` 和 `--model` flag 覆盖 |
| `CLAUDE_CODE_SUBAGENT_MODEL` | sub-agent 任务使用的模型，未设置时 sub-agent 用主 session 同一模型 |
| `CLAUDE_CODE_AUTO_MODE_MODEL` | auto 模式下使用的模型，未指定时默认用主 session 模型 |


**行为开关**

| 变量 | 说明 |
| --- | --- |
| `CLAUDE_CODE_REMOTE` | 设为 `1` 或 `true` 启用远程/容器模式：延长 API 超时（120s vs 300s）、压制交互弹窗、调整输出格式 |
| `CLAUDE_CODE_SIMPLE` | 设为 `1` 或 `true`（或传 `--bare`）进入 bare mode：跳过 hooks、LSP 集成、plugin 同步、skill 目录遍历、署名、后台预取，以及所有 keychain/credentials 读取。认证必须通过 `ANTHROPIC_API_KEY` 或 `apiKeyHelper` 提供。适合轻量脚本化使用 |
| `DISABLE_AUTO_COMPACT` | 设为 `1` 或 `true` 禁用自动 context compaction，即使接近模型的 context 上限也不压缩 |
| `DISABLE_AUTOUPDATER` | 禁用原生安装渠道下的自动更新，适合企业镜像或版本锁定环境 |
| `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS` | 设为 `1` 或 `true` 禁用后台任务执行，`run_in_background` 参数从 Bash 和 PowerShell tool 里移除 |
| `CLAUDE_CODE_DISABLE_THINKING` | 设为 `1` 或 `true` 对所有 API 调用禁用 extended thinking |
| `CLAUDE_CODE_DISABLE_AUTO_MEMORY` | 设为 `1` 或 `true` 禁用自动 memory；设为 `0` 或 `false` 显式启用（在 bare mode 或 remote mode 等本会禁用它的条件下也能开启）|
| `CLAUDE_CODE_DISABLE_CLAUDE_MDS` | 设为 `1` 或 `true` 完全禁用所有 `CLAUDE.md` memory 文件加载，包括 CLAUDE.local.md 和 `.claude/rules/*.md` |
| `CLAUDE_CODE_NO_EXTERNAL_REQUESTS` | 设为 `1` 或 `true` 压制分析、遥测及其他非必要网络请求 |
| `CLAUDE_CODE_LOAD_CLAUDE_MD_FROM_DIRS` | 设为 `1` 或 `true` 从通过 `--add-dir` 添加的目录加载 `CLAUDE.md` 文件 |
| `CLAUDE_CODE_RESET_CWD` | 设为 `1` 或 `true` 在每次 Bash 命令后将工作目录重置回原始项目根，防止一条命令影响后续命令的 CWD |

```bash
export CLAUDE_CODE_REMOTE=1
export DISABLE_AUTO_COMPACT=1
```


**资源限制**

| 变量 | 说明 |
| --- | --- |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | 覆盖每次 API 响应的最大输出 token 数，未设置时用模型默认值，适合控制自动化工作流的成本 |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | 覆盖最大 context 窗口大小，设置后 Claude Code 用此值替代模型报告的 context 上限 |
| `BASH_MAX_OUTPUT_LENGTH` | Bash 命令输出的最大捕获字符数，超出被截断，防止大量命令输出消耗 context |
| `API_TIMEOUT_MS` | 覆盖 API 请求超时（毫秒），标准模式默认 300000ms（5 分钟），remote 模式默认 120000ms（2 分钟）|

```bash
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096
export BASH_MAX_OUTPUT_LENGTH=50000
export API_TIMEOUT_MS=60000
```


**遥测与可观测性**

| 变量 | 说明 |
| --- | --- |
| `CLAUDE_CODE_ENABLE_TELEMETRY` | 设为 `1` 或 `true` 启用 OpenTelemetry 的 trace、metric 和 log 导出，需额外配置 OTEL 相关环境变量 |
| `CLAUDE_CODE_JSONL_TRANSCRIPT` | Claude Code 写入 JSONL session transcript 的文件路径，每行是代表一个对话事件的 JSON 对象 |

```bash
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otel.example.com"
export CLAUDE_CODE_JSONL_TRANSCRIPT="/tmp/session.jsonl"
```


**Node.js 运行时**

| 变量 | 说明 |
| --- | --- |
| `NODE_OPTIONS` | 传给运行时的标准 Node.js 选项，Claude Code 读取它检测如 `--max-old-space-size` 等 flag 并相应调整行为 |

```bash
export NODE_OPTIONS="--max-old-space-size=4096"
```

避免把 `NODE_OPTIONS` 设为包含代码执行 flag 的值，Claude Code 会对此类值要求确认再运行 Bash 命令。


**宿主平台覆盖**

`CLAUDE_CODE_HOST_PLATFORM`：覆盖分析用的宿主平台报告。接受 `"win32"`、`"darwin"` 或 `"linux"`，在容器里运行时（`process.platform` 报告容器 OS 但实际宿主不同）有用。


**云平台区域覆盖**

Claude Code 支持按模型覆盖 Vertex AI 区域，把特定模型路由到不同区域：

| 模型前缀 | 环境变量 |
| --- | --- |
| `claude-haiku-4-5` | `VERTEX_REGION_CLAUDE_HAIKU_4_5` |
| `claude-3-5-haiku` | `VERTEX_REGION_CLAUDE_3_5_HAIKU` |
| `claude-3-5-sonnet` | `VERTEX_REGION_CLAUDE_3_5_SONNET` |
| `claude-3-7-sonnet` | `VERTEX_REGION_CLAUDE_3_7_SONNET` |
| `claude-opus-4-1` | `VERTEX_REGION_CLAUDE_4_1_OPUS` |
| `claude-opus-4` | `VERTEX_REGION_CLAUDE_4_0_OPUS` |
| `claude-sonnet-4-6` | `VERTEX_REGION_CLAUDE_4_6_SONNET` |
| `claude-sonnet-4-5` | `VERTEX_REGION_CLAUDE_4_5_SONNET` |
| `claude-sonnet-4` | `VERTEX_REGION_CLAUDE_4_0_SONNET` |

```bash
# 把 Opus 4 路由到特定 Vertex 区域
export VERTEX_REGION_CLAUDE_4_0_OPUS="us-central1"
```

默认 Vertex 区域由 `CLOUD_ML_REGION` 控制（默认 `us-east5`）。


**AWS Credentials**

Bedrock 访问时，Claude Code 遵守标准 AWS credential 环境变量：`AWS_REGION`（Bedrock API 调用的 AWS 区域，回退到 `AWS_DEFAULT_REGION`，然后默认 `us-east-1`）。


**在所有 session 里设置环境变量**

可以在 settings 文件的 `env` 字段里设置适用于所有 Claude Code session 的环境变量，而不是在 shell profile 里设置：

```json
{
  "env": {
    "DISABLE_AUTO_COMPACT": "1",
    "BASH_MAX_OUTPUT_LENGTH": "30000"
  }
}
```

---

## 把三组文档放在一起看

把 `Core Concepts`、`Guides`、`Configuration` 三组页面放在一起看，Claude Code 的系统设计几个特点很清晰。

**它优先解决"稳定协作"，不追求"全自动神迹"**。文档反复强调的是：先把上下文组织好，先把规则写清楚，先把权限控制住，先让 tool 边界明确，先让配置和记忆可分层覆盖。这是很典型的工程路线，不浪漫，但有效。

**它把"运行环境"纳入了 prompt 的一部分**。当前日期、git 状态、`CLAUDE.md`、settings、environment variables 合起来构成 Claude Code 的真实工作上下文。这解释了为什么同样一个模型，在 Claude Code 里会比在普通聊天框里更像"能干活的同事"——不是模型突然懂了更多，而是它看到的世界变得结构化了。

**它把扩展分成了三层**：`CLAUDE.md` / skills 告诉 agent 应该怎么做事；hooks 在关键事件点插入流程控制；MCP 把外部能力真正接成 tool。不同问题在不同层解决，系统就稳。

**它默认接受"长时间协作"是常态**。transcript、resume、compaction、sub-agent、后台任务，这些能力说明 Claude Code 并不按"一问一答"来设计，而是默认你会和它一起做持续任务，甚至跨多轮、跨会话、跨目录地协作。

从这个意义上说，Claude Code 更像一个长期工作的 agent terminal，而不是一个临时回答问题的模型壳。

---

## 参考链接

- [Claude Code 首页](https://www.mintlify.com/VineeTagarwaL-code/claude-code)
- [Set up Claude Code](https://docs.claude.com/en/docs/claude-code/getting-started)
- [How Claude Code works](https://www.mintlify.com/VineeTagarwaL-code/claude-code/concepts/how-it-works)
- [Memory and context (CLAUDE.md)](https://www.mintlify.com/VineeTagarwaL-code/claude-code/concepts/memory-context)
- [Permissions](https://www.mintlify.com/VineeTagarwaL-code/claude-code/concepts/permissions)
- [Tools](https://www.mintlify.com/VineeTagarwaL-code/claude-code/concepts/tools)
- [Authentication](https://www.mintlify.com/VineeTagarwaL-code/claude-code/guides/authentication)
- [MCP servers](https://www.mintlify.com/VineeTagarwaL-code/claude-code/guides/mcp-servers)
- [Hooks](https://www.mintlify.com/VineeTagarwaL-code/claude-code/guides/hooks)
- [Skills](https://www.mintlify.com/VineeTagarwaL-code/claude-code/guides/skills)
- [Multi-agent workflows](https://www.mintlify.com/VineeTagarwaL-code/claude-code/guides/multi-agent)
- [Settings](https://www.mintlify.com/VineeTagarwaL-code/claude-code/configuration/settings)
- [CLAUDE.md](https://www.mintlify.com/VineeTagarwaL-code/claude-code/configuration/claudemd)
- [Environment variables](https://www.mintlify.com/VineeTagarwaL-code/claude-code/configuration/environment-variables)
