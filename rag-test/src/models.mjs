// ============================================
// 模型配置的唯一出口。每个脚本开头只写一行：
//   import { model, embeddings } from "./models.mjs";
// 只用对话就只 import model，只用向量化就只 import embeddings。
//
// 为什么单独放一个文件：这段配置以前在每个脚本里重复一遍，
// 只要 embeddings 的 baseURL 跟着 OPENAI_BASE_URL 走就会 404
// （本地网关 7863 只有对话路由，没有 /embeddings，报错却是 MODEL_NOT_FOUND）。
// 收口到这里，同类错误最多只犯一次。
//
// .env 由这里统一加载；注意 dotenv 是按「执行 node 时的当前目录」找 .env，
// 所以脚本必须在 rag-test 目录下运行（不是 src 目录）。
// ============================================
import "dotenv/config";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";

// 对话 / 推理 → .env 的 OPENAI_*（本机指向 127.0.0.1:7863 网关）
export const model = new ChatOpenAI({
  temperature: 0,
  model: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
});

// 向量化 → 两种 .env 形态都支持：
//   A. 两套配置：EMBEDDING_*（硅基流动 bge-m3）
//   B. 单套配置：OPENAI_* + EMBEDDINGS_MODEL_NAME（如阿里云百炼，chat 和 embedding 共用
//      https://dashscope.aliyuncs.com/compatible-mode/v1）
// 单独配 EMBEDDING_* 的原因：本地网关 7863 没有 /embeddings 路由，
// 直接沿用 OPENAI_BASE_URL 会 404。所以优先用 EMBEDDING_*，缺失时才回落。
const embeddingBaseURL = process.env.EMBEDDING_BASE_URL ?? process.env.OPENAI_BASE_URL;

if (
  process.env.EMBEDDING_BASE_URL == null &&
  process.env.OPENAI_BASE_URL?.includes("127.0.0.1")
) {
  throw new Error(
    "EMBEDDING_BASE_URL 未配置，且 OPENAI_BASE_URL 指向本地网关（没有 /embeddings 路由）。" +
      "请在 .env 里补 EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL。",
  );
}

export const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.EMBEDDING_API_KEY ?? process.env.OPENAI_API_KEY,
  model: process.env.EMBEDDING_MODEL ?? process.env.EMBEDDINGS_MODEL_NAME,
  configuration: {
    baseURL: embeddingBaseURL,
  },
});
