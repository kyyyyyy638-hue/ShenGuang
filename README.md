# ShenGuang

跟随《[神说要有光] AI Agent 开发》系列教程的动手实践仓库。全部代码为 JavaScript（Node + LangChain.js），按教程篇目分目录存放，每一篇都能独立运行。

教程源：[balabilibilo/agent-learning](https://github.com/balabilibilibo/agent-learning)

---

## 目录与篇目对照

| 目录 | 教程篇目 | 内容 |
|---|---|---|
| `tool-test/` | 02 Tool | 让大模型自动调用工具读文件 |
| | 03 mini cursor | 自己实现 agent 主循环（4 个工具 + 多轮调用） |
| | 04 MCP | 写 MCP server / 用 LangChain 接 MCP |
| | 05 高德 MCP | 接高德地图远程 MCP，带 QPS 限流提示 |
| `rag-test/` | 06 RAG | 文档向量化 + 内存向量库语义检索 |
| | 07 loader/splitter | Cheerio 加载网页 + 文本切分 |
| | 08 Splitter | 各类 splitter 行为对比、tiktoken 计费 |
| `milvus/` | 09 Milvus | Docker 起 Milvus，建集合、插数据、相似度检索 |
| | 10 Milvus + RAG | 检索结果拼 prompt，交给大模型生成回答 |

> 01「学什么」是概念篇，无代码。

---

## 学习进度

| 篇目 | 内容 | 完成日期 | 状态 |
|---|---|---|---|
| 01 | 学什么（概念篇） | — | 已读 |
| 02 | Tool：让模型自动调工具读文件 | 2026-09-20 | 完成 |
| 03 | mini cursor：手写 agent 主循环 | 2026-09-23 | 完成 |
| 04 | MCP：写 server、接 client | 2026-09-23 | 完成 |
| 05 | 高德 MCP + 浏览器 MCP | 2026-09-24 | 完成 |
| 06 | RAG：文档向量化与语义检索 | 2026-10-13 | 完成 |
| 07 | loader / splitter：网页加载与切分 | 2026-10-13 | 完成 |
| 08 | 各类 Splitter 对比 + tiktoken | 2026-10-13 | 完成 |
| 09 | Milvus：建集合、插入、相似度检索 | 2026-10-20 | 完成 |
| 10 | Milvus + RAG 全流程 | 2026-10-20 | 完成 |
| 10+ | 补充 upsert（更新）与 delete（删除） | 2026-10-29 | 完成 |
| 11 | Memory | — | 未开始 |
| 12 | 结构化输出 | — | 未开始 |

> 日期依据本地文件修改时间与 git 提交记录推断，若与实际学习日期不符，直接改这一列即可。

---

## 环境要求

| 项 | 版本 | 说明 |
|---|---|---|
| Node.js | v24.14.1（实测） | 需要 20+，因为用到原生 ESM 与 `TextDecoder` |
| pnpm | 11.22.0（实测） | 也可用 npm，但 lockfile 是 pnpm 的 |
| Docker | 任意较新版本 | 仅 `milvus/` 需要 |

### Windows 用户必看：pnpm 必须用 hoisted 模式

pnpm 默认的 `isolated` 布局会在 `node_modules/` 下创建 **Windows Junction**。在中文路径下这套机制不稳定，症状是：包明明在 `.pnpm/` 里，`node` 却报 `ERR_MODULE_NOT_FOUND`，`dir` 进去还报 `File Not Found` 或 `untrusted mount point`。

解决方式是在项目里固定 `hoisted` 布局（生成真实目录，不用 junction）：

```yaml
# pnpm-workspace.yaml
nodeLinker: hoisted
```

`milvus/milvus-test/` 已经这样配了。`rag-test/` 尚未配置，见文末「已知问题」。

---

## 快速开始

每个子项目都需要自己创建 `.env`（**已被 gitignore，不会提交**），变量说明见下一节。

### tool-test —— 工具调用与 agent 循环

```powershell
cd tool-test
pnpm install

node ./src/self-test-tool.mjs        # 不需要 key，先验证工具本身
bash test-api.sh                     # curl 直连，隔离 LangChain 排查
node ./src/hello-langchain.mjs       # 确认模型连通
node ./src/tool-file-read.mjs        # 观察 tool_calls（不循环）
node ./src/tool-file-read-loop.mjs   # 完整 agent 循环
node ./src/min-cursor.mjs            # 自己实现的 mini cursor
```

更多细节（GLM-5.3 的 `thinking` 参数适配等）见 [`tool-test/README.md`](tool-test/README.md)。

### rag-test —— 语义检索

```powershell
cd rag-test
pnpm install

node ./src/hello-rag.mjs              # 内存向量库 + 检索 + 生成
node ./src/check-score.mjs            # 手工算余弦相似度，校验分数口径
node ./src/loader-and-splitter.mjs    # 网页加载 + 切分
node ./src/CharacterTextSplitter-test.mjs
node ./src/TokenTextSplitter-test.mjs
node ./src/tiktoken-test.mjs          # token 计数与切分成本
```

### milvus —— 向量数据库

先起数据库（首次启动会拉镜像，需要几分钟）：

```powershell
cd milvus
docker compose -f milvus-standalone-docker-compose.yaml up -d
```

确认 `19530` 端口可用后：

```powershell
cd milvus/milvus-test
pnpm install

node ./src/insert.mjs   # 建集合 + 插入 5 条日记
node ./src/rag.mjs      # 检索 + 交给大模型回答
```

`insert.mjs` 可重复运行：检测到集合已存在会先删后建，不会主键冲突。

停止并清理数据：

```powershell
cd milvus
docker compose -f milvus-standalone-docker-compose.yaml down
```

---

## 环境变量

三个子项目用的模型服务**不一样**，不要互相复制 `.env`。

### tool-test（智谱 GLM-5.3）

```ini
OPENAI_API_KEY=<智谱 API Key>
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
MODEL_NAME=glm-5.3
REASONING_EFFORT=low
```

Key 获取：https://open.bigmodel.cn/ → API Keys（**只在创建时完整展示一次**）。

### rag-test / milvus-test（对话与向量化分开）

```ini
# 对话：本地网关（或任意 OpenAI 兼容端点）
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=http://127.0.0.1:7863/v1
MODEL_NAME=cn:deepseek-v4.1-flash

# 向量化：必须单独配，不能用对话端点
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-m3
```

**为什么要拆成两套**：对话网关通常只有 `/chat/completions`，没有 `/embeddings`。如果把 embedding 也指向它，会在第一步 404，而报错信息完全看不出是这个原因。

`src/models.mjs` 是模型配置的唯一出口，所有脚本都从它 import，不各自 new。若只用一个百炼 Key 同时跑对话和向量化，它也会自动回落到 `OPENAI_*` + `EMBEDDINGS_MODEL_NAME`。

---

## 踩坑记录

按遇到顺序整理，都是实测结论。

### 1. pnpm + 中文路径 → junction 失效

症状：`ERR_MODULE_NOT_FOUND: Cannot find package 'dotenv'`，但 `node_modules/dotenv` 明明存在。

排查：`fsutil reparsepoint query` 显示 junction 的重解析数据完全正确，`cmd /c dir` 却报 `File Not Found`。`pnpm install` 一直说 `Already up to date`，因为它只信 `.modules.yaml` 的记录，不校验链接能否走通。

结论：改用 `nodeLinker: hoisted`，彻底不创建 junction。

### 2. 删除 `node_modules` 无效（工具环境特定）

在 CodeBuddy 沙箱里执行删除会失败，原因是 IDE 通过 `NODE_OPTIONS` 注入了 `node-language-shim.cjs`，它把删除重定向到回收站程序，而该程序超时。

绕过方式：执行命令前清空该变量。

```powershell
$env:NODE_OPTIONS=""
node -e "require('fs').rmSync('node_modules',{recursive:true,force:true})"
```

自己的终端一般没有这个问题，`NODE_OPTIONS` 只在 IDE 的进程里。

### 3. Milvus SDK 3.x 的 `flush` 参数是复数

```js
await client.flush({ collection_names: [COLLECTION_NAME] });  // 正确
await client.flush({ collection_name: COLLECTION_NAME });     // 报 collection_name property is missing
```

报错文字有误导性：它说的是 `collection_name` 缺失，实际是参数名应为 `collection_names` 且必须是数组。

### 4. 小数据量用 FLAT，别抄 IVF_FLAT

教程用 `IVF_FLAT + nlist: 1024`，但只有 5 条数据时会触发 `nlist should be less than row count` 警告。几千条以内用 `IndexType.FLAT`（暴力精确比对）即可，无需调参。

### 5. 插入后必须 flush，查询前必须 load

- 不 `flush`：刚插的数据可能立刻查不到
- 不 `loadCollection`：搜索直接报 collection not loaded
- `load` 是幂等的，重复调用不会报错

### 6. `VECTOR_DIM` 必须和模型的实际输出一致

`BAAI/bge-m3` 和 `text-embedding-v3` 默认都是 1024 维。换 768 维模型时集合定义要跟着改，否则插入失败。`insert.mjs` 里加了运行时校验，维度不符会直接抛出可读的报错。

### 7. Windows 上 GBK / UTF-8 编码混流

`child_process` 抓 cmd 输出时，`dir` 等内部命令按控制台代码页（GBK）编码，而 `node` / `pnpm` 等外部程序写 UTF-8，两种流混在一起，单一解码器必然有一半乱码。解法是命令前加 `chcp 65001 > nul && `，把整条管道统一成 UTF-8。

---

## 仓库约定

- `.env`、`node_modules/`、`milvus/volumes/`、`*.log` 均已忽略，不提交
- `milvus/volumes/` 是 Milvus 的运行时数据（etcd / minio / WAL），clone 后由 docker compose 自动生成
- 依赖声明与 lockfile 要一起提交

## 已知问题

`rag-test/node_modules` 在当前机器上已损坏（`@langchain` 目录为空、`.pnpm` 下有残缺条目），导致该目录下所有脚本无法运行。修复方式：

```powershell
cd rag-test
$env:NODE_OPTIONS=""
node -e "require('fs').rmSync('node_modules',{recursive:true,force:true})"
# 在 pnpm-workspace.yaml 里加上 nodeLinker: hoisted 后
pnpm install
```

`milvus/milvus-test/` 不受影响，可正常运行。
