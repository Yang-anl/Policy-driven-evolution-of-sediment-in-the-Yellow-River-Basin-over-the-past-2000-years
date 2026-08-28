library(pls)

# 1. 读取数据
data <- read.csv("C:/Users/anley/Desktop/new_输沙量.csv")
head(data)

# 2. 对部分变量取log10
data$precipitation <- log10(data$precipitation)
data$corpland <- log10(data$corpland)
data$vegetation <- log10(data$vegetation)
data$sediment <- log10(data$sediment)

# 3. 标准化整个数据框并转换为数据框
scaled_data <- scale(data)
scaled_data <- as.data.frame(scaled_data)

# 4. 提取自变量和因变量
X <- scaled_data[, c("policy", "corpland", "vegetation", 
                     "precipitation", "temperature")]
y <- scaled_data$sediment

# 5. 构建PLSR模型（ncomp取自变量个数，可后续调整）
plsr_model <- plsr(y ~ as.matrix(X), ncomp = ncol(X), validation = "CV")

# 6. 查看模型结果
summary(plsr_model)

# 7. 提取PLSR系数
coef_plsr <- as.vector(coef(plsr_model, ncomp = ncol(X)))

# 8. 计算变量贡献率
beta_contrib_plsr <- data.frame(
  Variable = colnames(X),
  Beta = abs(coef_plsr),  # 绝对值
  Contribution_Pct = abs(coef_plsr) / sum(abs(coef_plsr)) * 100
)

# 9. 按贡献率排序
beta_contrib_plsr <- beta_contrib_plsr[order(-beta_contrib_plsr$Beta), ]
beta_contrib_plsr
