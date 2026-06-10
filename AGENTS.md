# 项目分层架构约束：禁止 Controller 直接访问数据库

## 强制规则

本项目采用 Controller -> Service -> Mapper/Repository 的分层架构。

### 1. Controller 层职责

Controller 层只允许负责：

- 接收 HTTP 请求；
- 参数校验；
- 调用 Service 层；
- 返回统一响应结果；
- 做少量 DTO / VO 转换。

Controller 层禁止包含任何数据库访问逻辑。

### 2. Controller 层禁止行为

在任何 `*Controller.java` 文件中，禁止出现以下行为：

- 禁止注入 Mapper，例如：

```java
@Autowired
private UserMapper userMapper;

@Resource
private OrderMapper orderMapper;