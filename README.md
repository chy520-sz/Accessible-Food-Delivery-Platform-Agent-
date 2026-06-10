# 无障碍外卖 Agent 完全体

## 2026-06-08 骑手端补充

项目组成新增：

```text
take-out/rider   # 骑手端 Vue 3 + Vite + TypeScript + Pinia + Element Plus
```

本地端口新增：

| 服务 | 地址 |
| --- | --- |
| 骑手端 | `http://localhost:5176` |

启动骑手端：

```powershell
cd take-out/rider
npm install
npm run dev
```

已有数据库升级需要执行：

```sql
source take-out/server/src/main/resources/sql/migrate_add_rider_portal.sql;
```

默认测试骑手账号：

| 入口 | 手机号 | 密码 | 说明 |
| --- | --- | --- | --- |
| 骑手端 | `13800000004` | `admin123` | 默认测试骑手 |

新增接口：

- `POST /api/auth/rider/register`
- `POST /api/auth/rider/login`
- `GET /api/rider/dashboard`
- `GET /api/rider/orders/available`
- `POST /api/rider/orders/{id}/accept`
- `GET /api/user/orders/current`
- `GET /api/rider/orders`
- `GET /api/rider/orders/{id}`
- `PUT /api/rider/orders/{id}/status`

骑手端工作台能力：

- 首页统计：今日配送收入、今日完成订单、进行中订单、超时订单、可接订单
- 可接订单池：展示未分配订单并支持手动接单
- 我的订单：支持状态筛选和超时筛选
- 订单详情：展示配送费、预约配送时间、下单/分配/取餐/送达/完成时间
- 配送费字段：`orders.delivery_fee`，默认 5.00

面向视障人群的自动化外卖点餐系统，包含外卖业务前后端、店铺端、管理端和语音 Agent 辅助服务。项目重点解决视障用户在浏览菜品、语音交互、加入购物车、下单确认、订单查询和评分推荐中的可访问性问题。

## 项目组成

```text
Agent完全体/
├── take-out/        # Java 后端 + 用户端 + 管理端 + 店铺端
├── take-out Agent/  # Python FastAPI 语音 Agent 服务
└── references/      # 项目知识库，记录架构、接口和历史决策
```

## 核心功能

- 用户端：无障碍点餐、菜品/套餐浏览、购物车、地址、订单、评分、语音引导和 Agent 对话。
- 店铺端：店铺账号登录、店铺工作台、本店菜品管理、本店套餐管理、本店订单和营业额统计。
- 管理端：超级管理员登录、全局查看用户/店铺/菜品/套餐/订单，菜品和套餐仅允许上下架。
- Java 后端：统一 REST API、JWT 鉴权、角色隔离、MySQL 数据持久化、Redis 缓存。
- Agent 服务：文本/语音对话、登录态同步、调用 Java 后端工具完成点餐和推荐。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 后端 | Java 17, Spring Boot 3.2.5, MyBatis-Plus, MySQL, Redis, JWT, BCrypt |
| 用户端 | Vue 3, Vite, TypeScript, Pinia, Element Plus |
| 管理端 | Vue 3, Vite, TypeScript, Pinia, Element Plus, ECharts |
| 店铺端 | Vue 3, Vite, TypeScript, Pinia, Element Plus, ECharts |
| Agent | Python, FastAPI, LangChain, DeepSeek 兼容 OpenAI 接口, edge-tts |

## 本地端口

| 服务 | 地址 |
| --- | --- |
| Java 后端 | `http://localhost:3000` |
| Agent 服务 | `http://localhost:8000` |
| 管理端 | `http://localhost:5173` |
| 店铺端 | `http://localhost:5174` |
| 用户端 | `http://localhost:5175` |

## 快速启动

### 1. 启动 Java 后端和前端

详细步骤见 [take-out/README.md](./take-out/README.md)。

常用命令：

```powershell
cd take-out/server
mvn spring-boot:run

cd ../client
npm install
npm run dev

cd ../admin
npm install
npm run dev

cd ../shop
npm install
npm run dev
```

### 2. 启动 Agent 服务

```powershell
cd "take-out Agent"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python main.py
```

启动前需要在 `.env` 中配置大模型 API Key；如需语音识别，还需要配置阿里云智能语音相关参数。

## 数据库

全新数据库使用：

```sql
source take-out/server/src/main/resources/sql/init.sql;
```

已有数据库升级时，按功能执行对应迁移脚本，例如：

```sql
source take-out/server/src/main/resources/sql/migrate_add_shop_portal.sql;
```

## 默认测试账号

| 入口 | 手机号 | 密码 | 说明 |
| --- | --- | --- | --- |
| 用户端 | `13900000000` | `123456` | 测试用户 |
| 管理端 | `13800000000` | `admin123` | 超级管理员 |
| 店铺端 | `13800000001` | `admin123` | 暖心食堂店铺账号 |
| 店铺端 | `13800000002` | `admin123` | 巷口面馆店铺账号 |
| 店铺端 | `13800000003` | `admin123` | 甜饮小站店铺账号 |

## 权限边界

- 管理端只能全局查看菜品/套餐，并只能控制上下架，不能新增、编辑、删除菜品或套餐。
- 店铺端只能访问当前登录店铺的数据，后端从 JWT 中读取 `shopId`，不信任前端传入的店铺 ID。
- 用户端和店铺端的登录态使用 `sessionStorage`，同一浏览器不同标签页登录不同账号时互不覆盖。
- 营业额统计仅店铺端可见，管理端不展示营业额。
- 营业状态不由管理端维护。

## 构建验证

```powershell
cd take-out/server
mvn -q -DskipTests clean compile

cd ../client
npm run build

cd ../admin
npm run build

cd ../shop
npm run build
```

Agent 语法检查：

```powershell
cd "take-out Agent"
python -m py_compile main.py agent.py tools.py prompts.py speech.py config.py
```

## 注意事项

- 不要提交真实的 `application.yml`、`.env`、Token、AccessKey 或 API Key。
- 如果出现 `Unknown column ...`，优先检查当前 MySQL 库是否执行了最新迁移脚本。
- 前端端口都启用了 `strictPort`，端口被占用时会直接失败，避免访问到错误前端。
