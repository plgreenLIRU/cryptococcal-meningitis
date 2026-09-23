library(tidyverse)
library(reticulate)

# import the load_data function from the Python script
source_python("utils.py")
data <- load_data("gold standard")

# Convert list of patients to tidy tibble
data_clean <- map_dfr(
  seq_along(data),
  function(i) {
    patient <- data[[i]]
    tibble(
      patient_id = i,
      time = patient$times,
      y = patient$y,
      log_y = patient$`log y`,
      data_type = patient$`data type`
    )
  }
) |>
  mutate(across(c(time, y, log_y), as.numeric))

data_clean |> 
  ggplot(aes(x = time, y = log_y, color = as.factor(patient_id))) +
  geom_line() +
  theme(legend.position = "none")

# filter only when there is a third LP
data_third_lp_only <- data_clean |>
  group_by(patient_id) |>
  filter(n() >= 3) |>
  ungroup()

data_third_lp_only |> 
  ggplot(aes(x = time, y = log_y, color = as.factor(patient_id))) +
  geom_line() +
  theme(legend.position = "none")

# Create patient-level groups from second/third log_y observations
patient_groups <- data_clean |>
  arrange(patient_id, time) |>
  group_by(patient_id) |>
  summarise(
    second_log_y = nth(log_y, 2, default = NA_real_),
    third_log_y = nth(log_y, 3, default = NA_real_),
    group = case_when(
      second_log_y < 0.5 ~ 1L,
      second_log_y > 0.5 & third_log_y < 0.5 ~ 2L,
      third_log_y > 0.5 ~ 3L,
      TRUE ~ 4L
    ),
    .groups = "drop"
  )

# Optional observation-level data with patient group attached
data_grouped <- data_clean |>
  left_join(patient_groups |> select(patient_id, group), by = "patient_id")

# Summarise mean and SD of log_y by group for first, second, and third LP
log_y_group_summary <- data_grouped |>
  arrange(patient_id, time) |>
  group_by(patient_id) |>
  mutate(lp_number = row_number()) |>
  ungroup() |>
  filter(lp_number %in% 1:3) |>
  group_by(group, lp_number) |>
  summarise(
    mean_log_y = mean(log_y, na.rm = TRUE),
    sd_log_y = sd(log_y, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) |>
  mutate(
    lp = case_when(
      lp_number == 1 ~ "first",
      lp_number == 2 ~ "second",
      lp_number == 3 ~ "third"
    )
  ) |>
  select(group, lp_number, lp, n, mean_log_y, sd_log_y) |>
  arrange(group, lp_number)

data_grouped |> 
  # keep only patients with three lps
  group_by(patient_id) |>
  filter(n() >= 3) |>
  ggplot(aes(x = time, y = log_y, color = as.factor(patient_id))) +
  geom_line() +
  facet_wrap(~ group) + 
  theme(legend.position = "none")

