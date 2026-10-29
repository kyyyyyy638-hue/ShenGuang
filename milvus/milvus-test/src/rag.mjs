// ============================================================
// 【这个文件干什么】
// 跑一遍完整的 RAG 流程：先检索，再让大模型基于检索结果回答。
//
// 【运行方式】必须在 milvus-test 目录下执行（.env 按当前目录找）：
//     cd E:\代码项目\ShenGuang\milvus\milvus-test
//     node src/rag.mjs
//
// 【前置条件】
//   1. 已经跑过 insert.mjs，集合 ai_diary 里有 5 条日记
//   2. Milvus 在 19530 监听
//   3. 对话模型和 Embedding 服务都可用
//
// 【什么是 RAG，一句话】
// 大模型不知道你的私人日记，直接问它只能瞎编。
// RAG = 先去数据库"翻"出最相关的几段原文，再把原文塞进问题一起问模型。
// 模型看着原文回答，就不会瞎编。流程：
//
//   问题 --[向量化]--> 向量 --[Milvus 相似度检索]--> 最相关的 N 篇日记
//                                                        ↓
//                                   [拼成上下文塞进 prompt] --> 大模型 --> 回答
//
// 【为什么不用 MySQL 的 LIKE 做关键词搜索】
// 问"我最近做了什么让我感到快乐的事情？"，日记里根本没有"快乐"这个词，
// 只有"心情愉快"、"很有成就感"。关键词搜索必须字面匹配，一个字对不上就搜不到；
// 向量检索比的是"意思像不像"，所以能命中。
// ============================================================

import "dotenv/config";

// 只需要 MilvusClient。DataType / IndexType 是建表用的，查询用不上；
// MetricType 也不必传 —— 建索引时已经声明了 COSINE，查询会沿用。
import { MilvusClient } from "@zilliz/milvus2-sdk-node";

// 【关键】模型统一从 models.mjs 拿，不要在这里自己 new。
// 教程写法是 new OpenAIEmbeddings({ configuration: { baseURL: OPENAI_BASE_URL } })，
// 在当前 .env 下会直接 404：OPENAI_BASE_URL 指向本地对话网关，
// 只有 /chat/completions，没有 /embeddings。
// models.mjs 已经把对话和向量化两套配置分开了，并支持单套配置回落。
import { model, embeddings } from "./models.mjs";

// 集合名，必须和 insert.mjs 里建的一致，否则查不到东西。
const COLLECTION_NAME = process.env.MILVUS_COLLECTION ?? "ai_diary";

// 建客户端。new 只是准备配置，真正连接在 main() 里的 connectPromise。
const client = new MilvusClient({
  address: process.env.MILVUS_ADDRESS ?? "localhost:19530",
});

/**
 * 第一步：把问题向量化，再去 Milvus 里找最像的 K 篇日记。
 *
 * 用 embedQuery（单个查询）而不是 embedDocuments（批量文档）：
 * 有些模型对这两种用途用不同前缀，混用会降低准确率。
 */
async function retrieveRelevantDiaries(question, k = 2) {
  // 问题也是一段文字，必须先变成 1024 维向量，
  // 才能和库里存的向量算相似度。
  const queryVector = await embeddings.embedQuery(question);

  // 搜索参数含义：
  //   vector        拿什么去比（刚算出来的问题向量）
  //   limit         返回最像的前 k 条
  //   output_fields 除 id 和 score 外，还要带回哪些字段
  const searchResult = await client.search({
    collection_name: COLLECTION_NAME,
    vector: queryVector,
    limit: k,
    output_fields: ["id", "content", "date", "mood", "tags"],
  });

  // searchResult 是响应对象，形如 { status: {...}, results: [...] }。
  // results 每项形如 { id, score, content, date, mood, tags } —— 字段是扁平的，
  // 不是塞在 entity 里（这点和部分 Milvus 2.x 客户端不同）。
  return searchResult.results ?? [];
}

/**
 * 第二步 + 第三步：用检索结果拼 prompt，让大模型回答。
 */
async function answerDiaryQuestion(question, k = 2) {
  console.log("=".repeat(80));
  console.log(`问题: ${question}`);
  console.log("=".repeat(80));

  // ---------- 检索 ----------
  console.log("\n【检索相关日记】");
  const retrievedDiaries = await retrieveRelevantDiaries(question, k);

  if (retrievedDiaries.length === 0) {
    console.log("未找到相关日记");
    return "抱歉，我没有找到相关的日记内容。";
  }

  // 打印出来，方便直观看到"向量检索到底命中了什么"。
  // ?. 是可选链：tags 万一不存在也不会让程序崩。
  retrievedDiaries.forEach((diary, i) => {
    console.log(`\n[日记 ${i + 1}] 相似度: ${Number(diary.score).toFixed(4)}`);
    console.log(`日期: ${diary.date}`);
    console.log(`心情: ${diary.mood}`);
    console.log(`标签: ${diary.tags?.join(", ")}`);
    console.log(`内容: ${diary.content}`);
  });

  // ---------- 拼上下文 ----------
  // 把 k 篇日记拼成一段纯文本。分隔符只是视觉分界，模型能看懂。
  const context = retrievedDiaries
    .map(
      (diary, i) => `[日记 ${i + 1}]
日期: ${diary.date}
心情: ${diary.mood}
标签: ${diary.tags?.join(", ")}
内容: ${diary.content}`,
    )
    .join("\n\n━━━━━\n\n");

  // ---------- 拼 prompt ----------
  // 这段模板做三件事：
  //   1. 给模型一个身份（温暖贴心的日记助手）
  //   2. 把检索到的原文放进去 —— 这就是 RAG 里的 R（Retrieval）的产物
  //   3. 用规则约束它不要瞎编（第 3 条：没有相关信息就如实说）
  const prompt = `你是一个温暖贴心的 AI 日记助手。基于用户的日记内容回答问题，用亲切自然的语言。

请根据以下日记内容回答问题：
${context}

用户问题: ${question}

回答要求：
1. 如果日记中有相关信息，请结合日记内容给出详细、温暖的回答
2. 可以总结多篇日记的内容，找出共同点或趋势
3. 如果日记中没有相关信息，请温和地告知用户
4. 用第一人称"你"来称呼日记的作者
5. 回答要有同理心，让用户感到被理解和关心

AI 助手的回答:`;

  // ---------- 调用大模型 ----------
  console.log("\n【AI 回答】");
  const response = await model.invoke(prompt);
  console.log(response.content);
  console.log("\n");

  return response.content;
}

async function main() {
  console.log("连接到 Milvus...");
  await client.connectPromise;
  console.log("✓ 已连接\n");

  // 集合必须处于 loaded 状态才能被搜索。
  // 之前 insert.mjs 加载过，但 Milvus 重启后可能需要重新加载；
  // load 是幂等的，重复调用不会报错。
  await client.loadCollection({ collection_name: COLLECTION_NAME });

  await answerDiaryQuestion("我最近做了什么让我感到快乐的事情？", 2);
}

// .catch 兜住任何一步的错误并给出人话提示；
// .finally 无论成败都关闭连接，否则 Node 进程会一直挂着不退出。
main()
  .catch((error) => {
    console.error("\n失败:", error.message);
    process.exitCode = 1;
  })
  .finally(async () => {
    await client.closeConnection();
  });
