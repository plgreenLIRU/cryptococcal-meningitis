library(tidyverse)
library(ggtext)

posterior_patient_samples_files <- list.files("sims",
                                              pattern = "points.csv",
                                              full.names = TRUE)
posterior_patient_samples <- map(posterior_patient_samples_files,
    ~ read_csv(.x)
  )
names(posterior_patient_samples) <- basename(tools::file_path_sans_ext(posterior_patient_samples_files))
posterior_patient_samples <- bind_rows(posterior_patient_samples, .id = "experiment_id")

posterior_patient_samples <- posterior_patient_samples %>% 
  mutate(
    growth_pattern = case_when(
      str_detect(experiment_id, "declining") ~ "declining",
      str_detect(experiment_id, "growth") ~ "growth",
      str_detect(experiment_id, "observed") ~ "observed",
      TRUE ~ "other"
  ),
    data_points = case_when(
      str_detect(experiment_id, "2_points") ~ "2",
      str_detect(experiment_id, "3_points") ~ "3",
      str_detect(experiment_id, "4_points") ~ "4",
      str_detect(experiment_id, "5_points") ~ "5",
      TRUE ~ "other"
    )
  )

population_posterior <- read_csv("sims/population_samples_temp.csv") %>%
  select(pd, pb)

model <- function(pd, c) {
  # returns pb given pd and c
  (c - 1 + pd) / (1 - pd)
}

pds <- seq(0, 0.99, by = 0.01)
cs <- seq(0.85, 1.15, by = 0.05)

pbs <- lapply(cs, function(c) {
  tibble(pds = pds, pb = model(pds, c))
})

names(pbs) <- cs
pbs <- bind_rows(pbs, .id = "c") %>% 
  mutate(c = as.numeric(c),
         c_text = paste0(-(1-c) * 100, "%"))

pbs <- pbs %>% 
  filter(pb >= 0 & pds >=0,
         !(pb > 1 | pds > 0.5))

# plot posterior samples using a 2d histogram
population_posterior %>% 
  ggplot(aes(x = pd, y = pb)) +
  geom_bin2d(bins = 100) +
  scale_fill_gradient(name = "Population \nposterior counts", low = "#d0e8f7", high = "#3a9fd6") +
  geom_line(data = filter(pbs, c_text != "0%"), aes(x = pds, y = pb, group = c_text), linetype = "dotted") +
  geom_line(data = filter(pbs, c_text == "0%"), aes(x = pds, y = pb), linetype = "solid", linewidth = 1.2) +
  geom_text(
    data = filter(pbs, c_text != "0%") |> group_by(c_text) |> slice_max(pds, n = 1),
    aes(x = pds, y = pb, label = c_text),
    hjust = -0.1, size = 3
  ) +
  geom_jitter(data = filter(posterior_patient_samples, data_points == "2"), 
              aes(x = pd, y = pb, colour = growth_pattern), 
              alpha = 0.3, size = 0.5) +
  scale_colour_manual(
    name = "Simulated scenario",
    values = c("declining" = "green", "growth" = "red", "other" = "orange"),
    labels = c("declining" = "Patient A", "growth" = "Patient B", "other" = "Other")
  ) +
  labs(
    x = "*p*~d~",
    y = "*p*~b~",
    title = "Population posterior samples with model line"
  ) +
  theme_minimal() +
  theme(
    axis.title.x = element_markdown(),
    axis.title.y = element_markdown()
  ) +
  xlim(0, 0.5) +
  ylim(0, 1)

