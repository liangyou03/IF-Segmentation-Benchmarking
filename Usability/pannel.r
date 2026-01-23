library(tidyverse)
library(funkyheatmap)
library(RColorBrewer)
library(ggplot2)

# ============================================================
# 1. 数据读取
# ============================================================
data <- read_csv("benchmark_table_wide_filled.csv")

# ============================================================
# 2. 计算 usab__time_rank 和 usab__dependence
# ============================================================
data <- data %>%
  mutate(
    # 基于runtime_seconds计算time_rank（越小rank越小）
    usab__time_rank = rank(runtime_seconds, ties.method = "average"),
    
    # 基于env_size_gb计算dependence score
    # 公式: 1.0 - 0.8 * (size - min_size) / (max_size - min_size)
    # 环境越小，分数越高
    usab__dependence = 1.0 - 0.8 * (env_size_gb - min(env_size_gb)) / 
                                    (max(env_size_gb) - min(env_size_gb)),
    usab__dependence = round(usab__dependence, 1)
  )

# ============================================================
# 3. 重新计算所有分数和排名
# ============================================================
acc_weight <- 0.8
usab_weight <- 0.2

data <- data %>%
  mutate(
    # acc__score (所有accuracy指标的总和)
    acc__score = acc__cell_brain__precision_iou50 + acc__cell_brain__recall_iou50 + 
                 acc__cell_brain__boundary_f1 + acc__heart_cell__recall_iou50 +
                 acc__nuclei__precision_iou50 + acc__nuclei__recall_iou50 + 
                 acc__nuclei__boundary_f1,
    
    # usab__score
    usab__score = (max(usab__time_rank) + 1 - usab__time_rank) / max(usab__time_rank) + 
                  usab__code_behavior + usab__dependence,
    
    # 标准化
    acc__score_std = (acc__score - min(acc__score)) / (max(acc__score) - min(acc__score)),
    usab__score_std = (usab__score - min(usab__score)) / (max(usab__score) - min(usab__score)),
    
    # 总分
    total_score = acc__score_std * acc_weight + usab__score_std * usab_weight,
    
    # 排名
    summary__accuracy_rank = rank(-acc__score),
    summary__usability_rank = rank(-usab__score),
    summary__overall_rank = rank(-total_score),
    usab__rank = rank(-usab__score)
  )

# 打印计算结果
cat("=== Calculated Metrics ===\n")
cat("\nusab__time_rank (based on runtime_seconds):\n")
print(data %>% 
      select(algorithm, runtime_seconds, usab__time_rank) %>% 
      arrange(usab__time_rank))

cat("\nusab__dependence (based on env_size_gb):\n")
print(data %>% 
      select(algorithm, env_size_gb, usab__dependence) %>% 
      arrange(desc(usab__dependence)))

cat("\nTop 5 Overall Rankings:\n")
print(data %>% 
      select(algorithm, summary__overall_rank, total_score, acc__score, usab__score) %>%
      arrange(summary__overall_rank) %>% 
      head(5))

# ============================================================
# 4. 归一化排名函数（用于可视化）
# ============================================================
normalize_rank_invert <- function(x) {
  if (all(is.na(x))) return(x)
  rng <- range(x, na.rm = TRUE)
  if (rng[1] == rng[2]) return(rep(1, length(x)))
  (rng[2] - x) / (rng[2] - rng[1])
}

# 归一化用于可视化
data_plot <- data %>%
  mutate(
    summary__overall_rank = normalize_rank_invert(summary__overall_rank),
    summary__accuracy_rank = normalize_rank_invert(summary__accuracy_rank),
    summary__usability_rank = normalize_rank_invert(summary__usability_rank),
    usab__rank = normalize_rank_invert(usab__rank),
    usab__time_rank = normalize_rank_invert(usab__time_rank)
  ) %>%
  arrange(desc(summary__overall_rank))

# ============================================================
# 5. 调色板
# ============================================================
my_palettes <- list(
  "Greys"   = brewer.pal(9, "Greys")[3:9],
  "Blues"   = brewer.pal(9, "Blues")[3:9],
  "Greens"  = brewer.pal(9, "Greens")[3:9],
  "Purples" = brewer.pal(9, "Purples")[3:9],
  "Oranges" = brewer.pal(9, "Oranges")[3:9]
)

# ============================================================
# 6. 自定义图例
# ============================================================
grey_colors <- brewer.pal(9, "Greys")[c(3, 4, 5, 6, 7)]
blue_colors <- brewer.pal(9, "Blues")[c(3, 4, 5, 6, 7)]
green_colors <- brewer.pal(9, "Greens")[c(3, 4, 5, 6, 7)]
orange_colors <- brewer.pal(9, "Oranges")[c(3, 4, 5, 6, 7)]
purple_colors <- brewer.pal(9, "Purples")[c(3, 4, 5, 6, 7)]

my_legends <- list(
  list(
    title = "Summary Rank",
    palette = "Greys",
    geom = "circle",
    labels = c("", "LOW", "", "", "HIGH  "),
    colour = grey_colors,
    size = c(0.15, 0.3, 0.5, 0.7, 0.85)
  ),
  list(
    title = "Brain Nuclei Accuracy",
    palette = "Greens",
    geom = "bar",
    labels = c(" 0.0", "", "0.5", "", "1.0 "),
    colour = green_colors
  ),
  list(
    title = "Brain Cell Accuracy",
    palette = "Blues",
    geom = "bar",
    labels = c(" 0.0", "", "0.5", "", "1.0 "),
    colour = blue_colors
  ),
  list(
    title = "Heart Cell Accuracy",
    palette = "Oranges",
    geom = "bar",
    labels = c(" 0.0", "", "0.5", "", "1.0 "),
    colour = orange_colors
  ),
  list(
    title = "Usability Score",
    palette = "Purples",
    geom = "circle",
    labels = c("", "  POOR", "", "", "GOOD  "),
    colour = purple_colors,
    size = c(0.15, 0.3, 0.5, 0.7, 0.85)
  )
)

# ============================================================
# 7. Column Info
# ============================================================
column_info <- tribble(
  ~id,                                 ~group,           ~name,            ~geom,        ~palette,      ~options,
  "algorithm",                         NA,               "Algorithm",      "text",       NA,            list(hjust = 0, width = 2.5),
  "group",                             NA,               "Group",          "text",       NA,            list(width = 4, color = "grey50"),
  
  # Summary
  "summary__overall_rank",             "summary",        "Overall",        "circle",     "Greys",       list(legend = FALSE),
  "summary__accuracy_rank",            "summary",        "Accuracy",       "circle",     "Blues",       list(legend = FALSE),
  "summary__usability_rank",           "summary",        "Usability",      "circle",     "Purples",     list(legend = FALSE),
  
  # Brain Nuclei Accuracy
  "acc__nuclei__precision_iou50",      "acc_nuclei",     "Precision",      "bar",        "Greens",      list(width = 2.5, legend = FALSE),
  "acc__nuclei__recall_iou50",         "acc_nuclei",     "Recall",         "bar",        "Greens",      list(width = 2.5, legend = FALSE),
  "acc__nuclei__boundary_f1",          "acc_nuclei",     "Boundary F1",    "bar",        "Greens",      list(width = 2.5, legend = FALSE),
  "brain_nuclei_AP",                   "acc_nuclei",     "AP",             "bar",        "Greens",      list(width = 2.5, legend = FALSE),

  # Brain Cell Accuracy
  "acc__cell_brain__precision_iou50",  "acc_brain",      "Precision",      "bar",        "Blues",       list(width = 2.5, legend = FALSE),
  "acc__cell_brain__recall_iou50",     "acc_brain",      "Recall",         "bar",        "Blues",       list(width = 2.5, legend = FALSE),
  "acc__cell_brain__boundary_f1",      "acc_brain",      "Boundary F1",    "bar",        "Blues",       list(width = 2.5, legend = FALSE),
  "brain_cell_AP",                     "acc_brain",      "AP",             "bar",        "Blues",       list(width = 2.5, legend = FALSE),

  # Heart Cell Accuracy
  "acc__heart_cell__recall_iou50",     "acc_heart",      "Recall",         "bar",        "Oranges",     list(width = 2.5, legend = FALSE),
  
  # Usability
  "usab__time_rank",                   "usab_merged",    "Time",           "circle",     "Purples",     list(legend = FALSE),
  "usab__code_behavior",               "usab_merged",    "Code Behav.",    "circle",     "Purples",     list(legend = FALSE),
  "usab__dependence",                  "usab_merged",    "Dependencies",   "circle",     "Purples",     list(legend = FALSE)
)

# ============================================================
# 8. Column Groups
# ============================================================
column_groups <- tribble(
  ~group,        ~palette,           ~level1,
  "summary",     "Greys",            "Summary",
  "acc_nuclei",  "Greens",           "Brain Nuclei",
  "acc_brain",   "Blues",            "Brain Cell",
  "acc_heart",   "Oranges",          "Heart Nuclei",
  "usab_merged", "Purples",          "Usability"
)

# ============================================================
# 9. 绘图
# ============================================================
g <- funky_heatmap(
  data = data_plot,
  column_info = column_info,
  column_groups = column_groups,
  palettes = my_palettes,
  legends = my_legends,
  scale_column = FALSE,
  position_args = position_arguments(
    col_annot_offset = 1.5, 
    expand_ymax = 0.3, 
    expand_ymin = 0.3
  )
)

# ============================================================
# 10. 保存
# ============================================================
g_final <- g +
  theme(
    text = element_text(family = "sans", size = 12),
    plot.margin = margin(t = 10, r = 10, b = 20, l = 10, unit = "pt"),
    legend.position = c(0.15, 0.3),
    legend.box = "horizontal"
  )

ggsave("benchmark_final.pdf", g_final, width = 17, height = 8)

# 保存计算后的数据
write_csv(data, "benchmark_table_calculated.csv")
cat("\n✓ Calculated data saved to: benchmark_table_calculated.csv\n")

g_final