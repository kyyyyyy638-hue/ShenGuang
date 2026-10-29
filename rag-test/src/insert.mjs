// ============================================
// 建集合 + 插入向量数据
// 运行：cd rag-test && node src/insert.mjs
//
// 向量化统一走 models.mjs 的 embeddings，不在这里重复配 baseURL。
// 之前这个文件自己 new 了一个 OpenAIEmbeddings 并绑定 OPENAI_BASE_URL，
// 而 OPENAI_BASE_URL 指向本地对话网关（只有 /chat/completions），
// 插入时会在 /embeddings 上 404 —— 这正是 models.mjs 当初要收口的原因。
// ============================================
import "dotenv/config";
import { embeddings } from "./models.mjs";
import {
  MilvusClient,
  DataType,
  MetricType,
  IndexType,
} from "@zilliz/milvus2-sdk-node";

const COLLECTION_NAME = process.env.MILVUS_COLLECTION ?? "ai_diary";
// 必须与 .env 里 EMBEDDING_MODEL 的实际输出维度一致：
//   BAAI/bge-m3          -> 1024
//   text-embedding-v3    -> 1024（默认）
const VECTOR_DIM = 1024;

const client = new MilvusClient({
  address: process.env.MILVUS_ADDRESS ?? "localhost:19530",
});

const diaryEntries = [
  {
    id: "diary_001",
    content: "今天天气很好，去公园散步了，心情愉快。看到了很多花开了，春天真美好。",
    date: "2026-01-10",
    mood: "happy",
    tags: ["生活", "散步"],
  },
  {
    id: "diary_002",
    content: "今天工作很忙，完成了一个重要的项目里程碑。团队合作很愉快，感觉很有成就感。",
    date: "2026-01-11",
    mood: "excited",
    tags: ["工作", "成就"],
  },
  {
    id: "diary_003",
    content: "周末和朋友去爬山，天气很好，心情也很放松。享受大自然的感觉真好。",
    date: "2026-01-12",
    mood: "relaxed",
    tags: ["户外", "朋友"],
  },
  {
    id: "diary_004",
    content: "今天学习了 Milvus 向量数据库，感觉很有意思。向量搜索技术真的很强大。",
    date: "2026-01-12",
    mood: "curious",
    tags: ["学习", "技术"],
  },
  {
    id: "diary_005",
    content: "晚上做了一顿丰盛的晚餐，尝试了新菜谱。家人都说很好吃，很有成就感。",
    date: "2026-01-13",
    mood: "proud",
    tags: ["美食", "家庭"],
  },
];

async function main() {
  await client.connectPromise;
  console.log(`已连接 Milvus: ${process.env.MILVUS_ADDRESS ?? "localhost:19530"}`);

  // 集合已存在就先删掉，保证脚本可以反复运行（否则 createCollection 会报已存在）
  const { value: exists } = await client.hasCollection({
    collection_name: COLLECTION_NAME,
  });
  if (exists) {
    await client.dropCollection({ collection_name: COLLECTION_NAME });
    console.log(`集合 ${COLLECTION_NAME} 已存在，已删除并重建`);
  }

  await client.createCollection({
    collection_name: COLLECTION_NAME,
    fields: [
      { name: "id", data_type: DataType.VarChar, max_length: 50, is_primary_key: true },
      { name: "vector", data_type: DataType.FloatVector, dim: VECTOR_DIM },
      { name: "content", data_type: DataType.VarChar, max_length: 5000 },
      { name: "date", data_type: DataType.VarChar, max_length: 50 },
      { name: "mood", data_type: DataType.VarChar, max_length: 50 },
      {
        name: "tags",
        data_type: DataType.Array,
        element_type: DataType.VarChar,
        max_capacity: 10,
        max_length: 50,
      },
    ],
  });
  console.log(`集合 ${COLLECTION_NAME} 已创建`);

  // 只有 5 条数据，用 FLAT 精确检索即可。
  // 教程里的 IVF_FLAT + nlist=1024 是为大数据集准备的，
  // 数据量远小于 nlist 时会触发 "nlist should be less than row count" 警告。
  await client.createIndex({
    collection_name: COLLECTION_NAME,
    field_name: "vector",
    index_type: IndexType.FLAT,
    metric_type: MetricType.COSINE,
  });
  console.log("向量索引已创建 (FLAT / COSINE)");

  await client.loadCollection({ collection_name: COLLECTION_NAME });
  console.log("集合已加载到内存");

  console.log(`\n开始向量化 ${diaryEntries.length} 条数据...`);
  const vectors = await embeddings.embedDocuments(
    diaryEntries.map((entry) => entry.content),
  );

  const actualDim = vectors[0]?.length;
  if (actualDim !== VECTOR_DIM) {
    throw new Error(
      `向量维度不匹配：模型返回 ${actualDim} 维，集合定义 ${VECTOR_DIM} 维。` +
        `请改 VECTOR_DIM 后重跑（集合已按新维度重建）。`,
    );
  }
  console.log(`向量化完成，维度 ${actualDim}`);

  const { insert_cnt } = await client.insert({
    collection_name: COLLECTION_NAME,
    data: diaryEntries.map((entry, i) => ({ ...entry, vector: vectors[i] })),
  });

  // flush 之后数据才可被检索到，否则刚插入的几条可能查不到
  // SDK 3.x 的 flush 参数是 collection_names 数组；传 collection_name 会报
  // "The `collection_name` property is missing."
  await client.flush({ collection_names: [COLLECTION_NAME] });

  console.log(`\n插入完成：${insert_cnt} 条 -> ${COLLECTION_NAME}`);
}

main()
  .catch((error) => {
    console.error("\n失败:", error.message);
    process.exit(1);
  })
  .finally(async () => {
    await client.closeConnection();
  });
