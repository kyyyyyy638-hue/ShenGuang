// ============================================================
// 【这个文件干什么】
// 演示向量库的「改」和「删」两个操作。
// 插入（insert）在 insert.mjs，查询（query / search）在 rag.mjs，
// 这个文件补上 update 和 delete，把增删改查凑齐。
//
// 【运行方式】必须在 milvus-test 目录下执行：
//     cd E:\代码项目\ShenGuang\milvus\milvus-test
//     node src/update.mjs
//
// 【前置条件】先跑过 insert.mjs，集合 ai_diary 里有 5 条数据。
//
// 【先理解一件事：Milvus 里没有 UPDATE 这个命令】
// 关系数据库有 UPDATE 语句，但向量库用的是 upsert（update + insert 的合成词）：
// 给一个主键 id，如果这条记录已存在就整行覆盖，不存在就新增一条。
// 所以「更新」和「新增」在 Milvus 里是同一个操作，区别只在于 id 是否已存在。
//
// 【最关键的坑：改了正文，必须重算向量】
// 向量是正文通过模型算出来的「语义指纹」。如果只改 content 不重算 vector，
// 库里就会出现「正文是 A、指纹是 B」的错位数据。
// 后果是：你搜 A 会命中这条，搜 B 反而搜不到 —— 而且不会报任何错，极难发现。
// ============================================================

import "dotenv/config";

// 只要 client。不建表，所以不需要 DataType / IndexType。
import { MilvusClient } from "@zilliz/milvus2-sdk-node";

// 向量化工具，改正文时必须用它把新内容重新变成向量。
import { embeddings } from "./models.mjs";

const COLLECTION_NAME = process.env.MILVUS_COLLECTION ?? "ai_diary";

const client = new MilvusClient({
  address: process.env.MILVUS_ADDRESS ?? "localhost:19530",
});

/**
 * 按主键查一条，用来对比操作前后的差异。
 *
 * filter 是 Milvus 的过滤表达式，语法类似 SQL 的 WHERE，
 * 但字符串要用双引号：id == "diary_001"。
 *
 * consistency_level: "Strong" 强制读最新数据。
 * 不写的话默认是 Bounded（有界延迟），刚改完立刻查可能读到旧值。
 */
async function fetchById(id) {
  const result = await client.query({
    collection_name: COLLECTION_NAME,
    filter: `id == "${id}"`,
    output_fields: ["id", "content", "mood", "date", "tags"],
    consistency_level: "Strong",
  });
  return result.data[0] ?? null;
}

/**
 * 查出当前集合里所有日记的 id，用来统计条数、确认删除效果。
 * 注意不要取 vector 字段 —— 1024 个数全拉回来既慢又没用。
 */
async function fetchAllIds() {
  const result = await client.query({
    collection_name: COLLECTION_NAME,
    filter: 'id like "diary%"',
    output_fields: ["id"],
    consistency_level: "Strong",
  });
  return result.data.map((row) => row.id).sort();
}

async function main() {
  await client.connectPromise;
  console.log(`已连接 Milvus: ${process.env.MILVUS_ADDRESS ?? "localhost:19530"}`);

  // 查询前集合必须处于 loaded 状态。load 是幂等的，重复调用不报错。
  await client.loadCollection({ collection_name: COLLECTION_NAME });

  // ==========================================================
  // 第 1 步：更新之前，先看一眼原始数据
  // ==========================================================
  console.log("\n【更新前】diary_001");
  const before = await fetchById("diary_001");
  if (!before) {
    console.error('没找到 diary_001，请先运行 node src/insert.mjs');
    return;
  }
  console.log(`内容: ${before.content}`);
  console.log(`心情: ${before.mood}`);
  console.log(`标签: ${before.tags?.join(", ")}`);

  // ==========================================================
  // 第 2 步：准备新数据，并重新计算向量
  // ==========================================================
  const updated = {
    id: "diary_001", // 主键不变 —— 相同 id 才会"覆盖"而不是"新增"
    content: "今天下雨了，窝在家里读完一本小说，心情很平静。",
    date: before.date, // 日期沿用原来的
    mood: "calm", // 心情从 happy 改成 calm
    tags: ["阅读", "独处"],
  };

  // 【重点】正文变了，向量必须跟着变。
  // 用 embedDocuments 传数组，拿回数组取第 0 个。
  console.log("\n【重新计算向量】");
  const [newVector] = await embeddings.embedDocuments([updated.content]);
  console.log(`新内容向量维度: ${newVector.length}`);

  // ==========================================================
  // 第 3 步：upsert 整行覆盖
  // ==========================================================
  // data 里必须给齐所有字段。因为 upsert 是"整行替换"，
  // 少给哪个字段，那个字段就变成空值 —— 不是"保持不变"。
  // （想只改某一个字段可以用 partial_update: true，但只能用于标量字段，
  //   向量字段仍然要整行给，所以这里直接用完整覆盖，逻辑最清楚。）
  const upsertResult = await client.upsert({
    collection_name: COLLECTION_NAME,
    data: [{ ...updated, vector: newVector }],
  });
  console.log(`\nupsert 完成，影响 ${upsertResult.upsert_cnt} 条`);

  // ==========================================================
  // 第 4 步：验证更新结果
  // ==========================================================
  await client.flush({ collection_names: [COLLECTION_NAME] });

  console.log("\n【更新后】diary_001");
  const after = await fetchById("diary_001");
  console.log(`内容: ${after.content}`);
  console.log(`心情: ${after.mood}`);
  console.log(`标签: ${after.tags?.join(", ")}`);

  // 顺手确认一下条数没变 —— upsert 覆盖已有记录，不会多出一条。
  console.log(`\n当前总条数: ${(await fetchAllIds()).length}（应为 5）`);

  // ==========================================================
  // 第 5 步：删除
  // ==========================================================
  // 两种写法都能删，这里用 filter 表达式，更灵活（可以一条语句删多条）：
  //   filter: 'id == "diary_005"'
  //   filter: 'mood in ["proud", "excited"]'   ← 按条件批量删
  // 另一种是直接给主键数组：{ ids: ["diary_005"] }
  console.log("\n【删除】diary_005");
  const deleteResult = await client.delete({
    collection_name: COLLECTION_NAME,
    filter: 'id == "diary_005"',
  });
  console.log(`delete 完成，删除 ${deleteResult.delete_cnt} 条`);

  // 删除同样需要 flush 才能立刻在查询里看到效果
  await client.flush({ collection_names: [COLLECTION_NAME] });

  // ==========================================================
  // 第 6 步：确认删除结果
  // ==========================================================
  const remaining = await fetchAllIds();
  console.log(`\n剩余 id: ${remaining.join(", ")}`);
  console.log(`剩余条数: ${remaining.length}（应为 4）`);

  const deleted = await fetchById("diary_005");
  console.log(`再查 diary_005: ${deleted ? "还能查到（异常）" : "已查不到 ✓"}`);

  // 说明：Milvus 的删除是「软删除」—— 记录会打上删除标记，
  // 查询立刻不可见，但磁盘上的数据要等后台 GC 才真正清理。
  // 所以 delete 之后空间不会马上释放，这是正常现象。
}

main()
  .catch((error) => {
    console.error("\n失败:", error.message);
    process.exitCode = 1;
  })
  .finally(async () => {
    await client.closeConnection();
  });
