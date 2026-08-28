#清除环境
library(brms)
#读取数据
data<-read.csv("C:/Users/anley/Desktop/data1.csv")
head(data)
data$corpland<-log10(data$corpland)
data$precipitation<-log10(data$precipitation)
data$population<-log10(data$population)
data<-scale(data)
#指定自相关项
mod1 <- bf(corpland~policy)
mod2 <- bf(population~policy)
mod3 <- bf(vegetation~policy+corpland+population)
mod4 <- bf(sediment~vegetation+precipitation+temperature)


#模型拟合
fit <- brm(mod1 +mod2+mod3 +mod4+
             set_rescor(FALSE),#是否重新评分（rescore）模型
           data=data,
           cores=4,#指定用于拟合模型的 CPU 核心数
           chains = 2#指定用于拟合模型的链数
)

summary(fit)
#Rhat 应接近 1（一般要求 < 1.05）
#结果解读
#Estimate表示路径系数
#基于可信区间 (Credible Intervals) 的显著性判断
#贝叶斯分析中，95% 可信区间（95% CI） 是最直接的显著性判断依据：
#区间不包含 0：参数有统计学意义（类似 p < 0.05）。
#区间包含 0：参数无统计学意义（类似 p ≥ 0.05）。
#示例解读：
#PH_TOC：Estimate = 0.54，95% CI = [0.26, 0.82]，区间不包含 0 → 显著（正向影响）。
#BD_TOC：Estimate = -0.18，95% CI = [-0.52, 0.16]，区间包含 0 → 不显著。
#ROCI_LOC：+Estimate = -0.79，95% CI = [-1.31, -0.31]，区间不包含 0 → 显著（负向影响）。
WAIC(fit)

#WAIC 的值越小，模型的拟合效果越好

# 计算后验预测 p 值
yrep <- posterior_predict(fit, resp = "sediment")  # 指定响应变量
data<-data.frame(data)
y_obs <- data$sediment
T_obs <- mean(y_obs)
T_rep <- apply(yrep, 1, mean)
p_value <- mean(T_rep > T_obs)
print(paste("Posterior predictive p-value:", p_value))
#后验预测 p 值 的解释与传统 p 值不同，接近 0.5 表示模型拟合良好，接近 0 或 1 表示存在问题

