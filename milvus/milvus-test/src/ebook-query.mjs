// ============================================================
// 【这个文件干什么】
// 在已导入的电子书里做语义检索：给一个问题，返回最相关的几块原文。
//
// 【运行方式】在 milvus-test 目录下：
//     cd E:\代码项目\ShenGuang\milvus\milvus-test
//     node src/ebook-query.mjs                        # 用默认问题
//     node src/ebook-query.mjs "孟宇和孙伟什么关系"     # 自己传问题
//
// 【前置条件】
//   1. 先跑过 ebook-writer.mjs，集合 ebook_chunks 里有数据
//   2. Milvus 在 19530 监听
//
// 【这一步只做检索，不做生成】
//   这里只回答"哪几块最相关"，还不会组织语言。
//   要让它像人一样回答，需要把这些块拼进 prompt 交给大模型 —— 那是下一步。
// ============================================================

import "dotenv/config";

import { MilvusClient } from "@zilliz/milvus2-sdk-node";

// 【关键】embeddings 必须从 models.mjs 拿，不要在这里自己 new。
// 教程写法是 configuration: { baseURL: process.env.OPENAI_BASE_URL }，
// 而当前 .env 里 OPENAI_BASE_URL 指向本地对话网关，只有 /chat/completions，
// 没有 /embeddings，一调用就 404。
import { embeddings } from "./models.mjs";

// 集合名必须和 ebook-writer.mjs 里建的一致，否则查不到东西
const COLLECTION_NAME = process.env.EBOOK_COLLECTION ?? "ebook_chunks";

// 返回最相关的几块
const TOP_K = 3;

// 问题来源：命令行第一个参数；没传就用默认问题
const QUESTION = process.argv[2] ?? "这个小说的主人公是谁";

const client = new MilvusClient({
  address: process.env.MILVUS_ADDRESS ?? "localhost:19530",
});

/**
 * 检索。
 *
 * 注意 output_fields 只列实际存在的字段：
 *   book         书名
 *   chapter_num  第几章
 *   chunk_index  章内第几块
 *   content      正文
 * 主键 id 不需要显式声明，搜索结果里默认就带。
 */
async function searchEbook(question, topK = TOP_K) {
  // 问题也要过同一个 embedding 模型，变成 1024 维向量才能比对
  const queryVector = await embeddings.embedQuery(question);

  // 不传 metric_type：建索引时已声明 COSINE，查询会沿用。
  // 两处传不一致反而可能报错。
  const result = await client.search({
    collection_name: COLLECTION_NAME,
    vector: queryVector,
    limit: topK,
    output_fields: ["book", "chapter_num", "chunk_index", "content"],
  });

  return result.results ?? [];
}

async function main() {
  console.log("连接到 Milvus...");
  await client.connectPromise;
  console.log("✓ 已连接\n");

  // 集合必须处于 loaded 状态才能搜索。
  // load 是幂等的，重复调用不会报错，所以不需要 try/catch 包一层。
  await client.loadCollection({ collection_name: COLLECTION_NAME });
  console.log("✓ 集合已加载\n");

  console.log("=".repeat(80));
  console.log(`问题: ${QUESTION}`);
  console.log("=".repeat(80));

  const results = await searchEbook(QUESTION);

  if (results.length === 0) {
    console.log("\n没有检索到任何内容。请确认已经跑过 ebook-writer.mjs。");
    return;
  }

  console.log(`\n找到 ${results.length} 块相关内容：\n`);

  results.forEach((item, i) => {
    // SDK 3.x 的 score 是字符串形式，直接调 toFixed 会报错，先转数字
    const score = Number(item.score).toFixed(4);

    console.log(`${i + 1}. [相似度: ${score}]`);
    console.log(`   出处: 《${item.book}》第 ${item.chapter_num} 章 · 第 ${item.chunk_index} 块`);
    console.log(`   内容: ${item.content.replace(/\s+/g, " ").slice(0, 150)}...`);

    // 相似度参考区间（COSINE，越接近 1 越像）：
    //   > 0.6   通常是真的相关
    //   0.4~0.6 有点关系，需要人工判断
    //   < 0.4   基本不相关，说明问题超出这本书的范围
    if (Number(item.score) < 0.4) {
      console.log("   ⚠ 相似度偏低，这条可能不相关");
    }
    console.log("");
  });

  // 顺便提示一下：检索只是"找材料"，还没到"回答问题"
  console.log("─".repeat(80));
  console.log("以上是检索到的原文片段。要让大模型基于这些内容生成回答，");
  console.log("需要把 content 拼进 prompt 再调用模型（下一步）。");
}

main()
  .catch((error) => {
    console.error("\n失败:", error.message);
    process.exitCode = 1;
  })
  .finally(async () => {
    // 不关闭连接的话，Node 进程会一直挂着不退出
    await client.closeConnection();
  });
