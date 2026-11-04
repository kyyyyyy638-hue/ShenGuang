// ============================================================
// 【这个文件干什么】
// 完整的 RAG：先在电子书里检索相关段落，再让大模型基于这些段落回答。
//
// 【运行方式】在 milvus-test 目录下：
//     cd E:\代码项目\ShenGuang\milvus\milvus-test
//     node src/ebook-rag.mjs                          # 默认问题
//     node src/ebook-rag.mjs "孙伟为什么要去郑州"      # 自己传问题
//
// 【和 ebook-query.mjs 的区别】
//   ebook-query.mjs  只检索，打印出最相关的原文片段，不做生成
//   ebook-rag.mjs    检索 + 生成，会给你一段通顺的自然语言回答
//
// 【RAG 的两个关键约束，都写在 prompt 里】
//   1. 只许依据给出的片段回答 —— 否则模型会用自己的知识瞎编
//   2. 片段里没有就如实说没有 —— 否则它会硬凑一个听起来合理的答案
// ============================================================

import "dotenv/config";

import { MilvusClient } from "@zilliz/milvus2-sdk-node";

// model 负责生成回答，embeddings 负责把问题变成向量。
// 都从 models.mjs 取，避免这文件里再配一遍 baseURL（配错就会 404）。
import { model, embeddings } from "./models.mjs";

const COLLECTION_NAME = process.env.EBOOK_COLLECTION ?? "ebook_chunks";

// 检索几块。太小可能漏掉答案，太大则 prompt 变长、且容易混入无关内容。
const TOP_K = 3;

const QUESTION = process.argv[2] ?? "小说一共有多少章";

const client = new MilvusClient({
  address: process.env.MILVUS_ADDRESS ?? "localhost:19530",
});

/**
 * 清理从 xhtml 里带出来的非法 HTML 实体残留。
 *
 * 这本书的 xhtml 里写了 `&“` 这种不合法的实体（& 后直接跟中文引号），
 * 标准解析器无法解码，只能原样保留。检索不受影响，
 * 但直接喂给大模型会让回答里冒出奇怪的 & 符号。
 */
function cleanText(text) {
  return text.replace(/&(?=[\u4e00-\u9fa5“”‘’])/g, "");
}

/**
 * 第一步：检索。和 ebook-query.mjs 里的逻辑一样。
 */
async function retrieve(question, topK = TOP_K) {
  const queryVector = await embeddings.embedQuery(question);

  const result = await client.search({
    collection_name: COLLECTION_NAME,
    vector: queryVector,
    limit: topK,
    output_fields: ["book", "chapter_num", "chunk_index", "content"],
  });

  return result.results ?? [];
}

/**
 * 第二步：把检索到的片段拼成上下文。
 *
 * 每块都标上出处（第几章第几块），有两个作用：
 *   - 模型能据此判断片段之间的关系
 *   - 回答出错时你能立刻翻回原文核对
 */
function buildContext(chunks) {
  return chunks
    .map(
      (chunk, i) => `[片段${i + 1}] 《${chunk.book}》第${chunk.chapter_num}章
${cleanText(chunk.content)}`,
    )
    .join("\n\n━━━━━\n\n");
}

/**
 * 第三步：拼 prompt。
 *
 * 措辞直接影响结果质量，这里做了四件事：
 *   1. 明确身份：基于小说原文的问答助手
 *   2. 给出唯一信息来源：下面的片段
 *   3. 禁止编造，并规定"没有就说没有"的表达
 *   4. 要求回答简短，不要复述整段原文
 */
function buildPrompt(question, context) {
  return `你是一个基于小说原文回答问题的助手。

请严格遵守以下规则：
1. 只依据下面提供的原文片段回答，不要使用片段之外的知识
2. 如果片段中没有相关信息，直接回答"原文中未找到相关内容"，不要猜测
3. 回答简洁自然，控制在 3 句话以内
4. 不要整段复述原文

原文片段：
${context}

问题：${question}

回答：`;
}

async function main() {
  console.log("连接到 Milvus...");
  await client.connectPromise;

  // 集合必须加载才能搜索，load 幂等
  await client.loadCollection({ collection_name: COLLECTION_NAME });

  console.log("=".repeat(80));
  console.log(`问题: ${QUESTION}`);
  console.log("=".repeat(80));

  // ---------- 检索 ----------
  console.log("\n【第一步：检索相关段落】");
  const chunks = await retrieve(QUESTION);

  if (chunks.length === 0) {
    console.log("没有检索到内容，请先运行 ebook-writer.mjs 导入数据。");
    return;
  }

  chunks.forEach((chunk, i) => {
    console.log(
      `${i + 1}. [${Number(chunk.score).toFixed(4)}] 第${chunk.chapter_num}章 第${chunk.chunk_index}块`,
    );
  });

  // ---------- 生成 ----------
  console.log("\n【第二步：交给大模型生成回答】");
  const prompt = buildPrompt(QUESTION, buildContext(chunks));

  const response = await model.invoke(prompt);

  console.log("\n" + "─".repeat(80));
  console.log(String(response.content).trim());
  console.log("─".repeat(80));

  // ---------- 出处 ----------
  console.log("\n【引用来源】");
  chunks.forEach((chunk, i) => {
    console.log(`${i + 1}. 《${chunk.book}》第 ${chunk.chapter_num} 章 第 ${chunk.chunk_index} 块`);
  });
}

main()
  .catch((error) => {
    console.error("\n失败:", error.message);
    process.exitCode = 1;
  })
  .finally(async () => {
    await client.closeConnection();
  });
