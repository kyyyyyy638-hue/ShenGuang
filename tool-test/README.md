# tool-test —— md2「从 Tool 开始：让大模型自动调工具读文件」实操

对应教程：https://github.com/balabilibilibo/agent-learning （02 篇）

## 文件与教程步骤的对应关系

| 文件 | 对应步骤 | 需要 API Key |
|---|---|---|
| `test-api.sh` | 【额外加的】curl 直连智谱，隔离掉 LangChain | 是 |
| `src/tools/read-file.mjs` | tool 定义（被下面几个 import） | 否 |
| `src/self-test-tool.mjs` | 【额外加的】不经过模型，先单测 tool | **否** |
| `src/hello-langchain.mjs` | Step 2 - 跑通裸聊天，确认模型连通 | 是 |
| `src/tool-file-read.mjs` | Step 5 - 只看 tool_calls，**不循环** | 是 |
| `src/tool-file-read-loop.mjs` | Step 6 - 完整 agent 循环 | 是 |

> 教程把 tool 定义和调用写在同一个文件里。这里拆成 `src/tools/read-file.mjs` 单独导出，
> 好处是能写 `self-test-tool.mjs` —— 不花 token、不需要 key 就能验证工具本身写对了没有。

## 运行顺序

```bash
# 0. 先填 key（见下面「填 key」一节）

# 1. 不需要 key，验证工具通路
node ./src/self-test-tool.mjs

# 2. 用 curl 直连智谱，确认 key / 模型名 / 账号额度都没问题（隔离掉 LangChain）
bash test-api.sh

# 3. 确认模型能连通
node ./src/hello-langchain.mjs

# 4. 看模型到底返回了什么（重点看 tool_calls）
node ./src/tool-file-read.mjs

# 5. 完整循环，让模型自动读文件并解释
node ./src/tool-file-read-loop.mjs
```

## 填 key（智谱 AI / BigModel）

编辑 `.env`，只需要填 `OPENAI_API_KEY` 这一行，其余已经配好：

```
OPENAI_API_KEY=<在这里填你的智谱 API Key>
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
MODEL_NAME=glm-5.3
REASONING_EFFORT=low
```

- Key 获取：https://open.bigmodel.cn/ → API Keys。**Key 只在创建时完整展示一次**，务必立即保存。
- `MODEL_NAME` 可选 `glm-5.3`（旗舰，输入 8 元 / 输出 28 元 每百万 token）
  或 `glm-5.3-flash`（便宜约 10 倍，输入 0.8 元）。先跑通建议用 flash。
- `.env` 里的变量名保留 `OPENAI_*` 前缀 —— 因为智谱兼容 OpenAI 协议，
  LangChain 读的就是这几个名字，改名要动代码，没必要。

## GLM-5.3 的两个适配点（不改会失败）

### 1. 必须显式传 `thinking.type = "enabled"`

GLM-5.3 **强制开启思考模式，关不掉**。官方迁移提示原文：不显式传
`thinking: { type: "enabled" }` 的话，**请求会直接失败**。
这个参数不能放 LangChain 顶层选项，要通过 `modelKwargs` 透传给 API：

```js
modelKwargs: {
  thinking: { type: 'enabled' },
  reasoning_effort: process.env.REASONING_EFFORT || 'low',
}
```

### 2. 去掉了 `temperature`

GLM-5.3 是强制思考的推理模型，官方示例不带 `temperature`。
教程原版写的是 `temperature: 0`（tool calling 场景求稳定），这里先去掉，
减少一个故障变量。将来若换回非推理模型再加回来。

### `reasoning_effort` 别用默认值

三档 `low / high / max`，**默认是 `max`** —— 简单任务也会产出很长的思考链，
而思考 token 是按输出价计费的。这些 demo 都是简单任务，`low` 足够；
复杂编码任务再往上调。

## 本机踩过的两个坑

### 1. pnpm 的 node_modules 顶层链接是空的

**现象**：`node ./src/self-test-tool.mjs` 报

```
Error [ERR_MODULE_NOT_FOUND]: Cannot find module
'...\node_modules\@langchain\core\tools'
```

**原因**：pnpm 默认 `isolated` 模式，用 junction 组织 `node_modules`。
在本机（Windows + 中文路径 `E:\代码项目\`）下顶层链接创建失败，留下空目录，
Node 读不到 `@langchain/core/package.json`，于是退化去直接找文件路径 → 找不到。
包内容在 `node_modules/.pnpm/` 里其实是完整的。

**验证方法**：直接看顶层能不能读到包清单

```bash
test -f node_modules/@langchain/core/package.json && echo "链接正常" || echo "链接坏了（空壳目录）"
# 别用 node -e "长脚本" 来验证 —— 本机沙箱会对长内联脚本发 SIGTERM，落成文件再跑
```

**解决**：改用 hoisted 模式。注意 **pnpm 11 不再从 `.npmrc` 读这个配置**（`pnpm config get node-linker`
返回 `undefined`），要写进 `pnpm-workspace.yaml`：

```yaml
nodeLinker: hoisted
```

或者临时命令行指定：

```bash
pnpm install --node-linker=hoisted
```

### 2. 占位 API Key 里带中文

**现象**：跑起来报

```
TypeError: Cannot convert argument to a ByteString because the character at
index 10 has a value of 25226 which is greater than 255.
```

字符码 25226 是「把」字。key 会被塞进 `Authorization` HTTP 头，而 HTTP 头只接受
latin-1 范围的字节，中文直接触发 `ByteString` 转换失败。

**教训**：报错信息里完全看不出是 key 的问题。`.env` 里的占位符一律用纯 ASCII。

## 教程代码的两个注意点（1.x 版本适配）

1. 教程写 `modelName:`，新版 `@langchain/openai@1.x` 用 `model:`。两者都能被接受，但 `model` 是当前推荐写法。
2. 完整循环里，教程只在**第一次** invoke 后 `messages.push(response)`，循环内部那份没 push。
   单轮工具调用看不出问题，但一旦模型需要连续调用 2 个以上工具，就会因为上一轮的
   AIMessage 缺失而报「ToolMessage 找不到对应的 tool_calls」。
   已在 `tool-file-read-loop.mjs` 里补上并标注。
