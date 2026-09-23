library(tidyverse)
library(ggtext)

posterior_patient_samples_files <- list.files(".",
                                              pattern = "points.csv",
                                              full.names = TRUE)
posterior_patient_samples <- map(posterior_patient_samples_files,
    ~ read_csv(.x)
  )
names(posterior_patient_samples) <- basename(tools::file_path_sans_ext(posterior_patient_samples_files))
posterior_patient_samples <- bind_rows(posterior_patient_samples, .id = "experiment_id")
posterior_patient_samples %>%
  filter(experiment_id == "declining_example_posterior_2_points") %>%
  pivot_longer(c("pd","pb"), names_to = 'param') %>%
  ggplot(aes(x = value, colour = param)) +
  geom_density()

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

posterior_patient_samples %>% 
  pivot_longer(c("pd","pb"), names_to = 'param') %>%
  ggplot(aes(x = value, colour = param)) +
  geom_density() +
  facet_grid(growth_pattern ~ data_points, scales = "free")

population_posterior <- read_csv("population_samples_temp.csv") %>%
  select(pd, pb)

population_mean <- colMeans(population_posterior)
population_cov <- cov(population_posterior)

# 95% joint region in 2D (roughly the 1-sigma ellipse)
ellipse_cutoff <- qchisq(0.95, df = 2)

# One analysis dataframe for all LP counts and both patterns.
analysis_df <- posterior_patient_samples %>%
  filter(growth_pattern %in% c("declining", "growth"), data_points %in% c("2", "3", "4", "5")) %>%
  mutate(
    data_points = factor(data_points, levels = c("2", "3", "4", "5")),
    d2 = mahalanobis(cbind(pd, pb), center = population_mean, cov = population_cov),
    inside_population_1sd_ellipse = d2 <= ellipse_cutoff
  )

# Summary table across all LP counts.
summary_table <- analysis_df %>%
  group_by(growth_pattern, data_points) %>%
  summarise(
    n_total = n(),
    n_in_population_1sd_ellipse = sum(inside_population_1sd_ellipse),
    pct_in_population_1sd_ellipse = 100 * mean(inside_population_1sd_ellipse),
    .groups = "drop"
  ) %>%
  arrange(data_points, growth_pattern)

summary_table

# Build ellipse path for plotting in (pd, pb) space.
ellipse_theta <- seq(0, 2 * pi, length.out = 200)
unit_circle <- cbind(cos(ellipse_theta), sin(ellipse_theta)) * sqrt(ellipse_cutoff)
eigen_decomp <- eigen(population_cov)
transform_matrix <- eigen_decomp$vectors %*% diag(sqrt(pmax(eigen_decomp$values, 0)))
ellipse_coords <- unit_circle %*% t(transform_matrix)
population_ellipse_df <- as_tibble(
  ellipse_coords + matrix(population_mean, nrow = nrow(ellipse_coords), ncol = 2, byrow = TRUE),
  .name_repair = "minimal"
)
colnames(population_ellipse_df) <- c("pd", "pb")

# Final plot: all LP counts, one analysis dataframe.
analysis_df %>%
  filter(data_points == "2") %>% 
  ggplot() +
  geom_path(
    data = population_ellipse_df,
    aes(x = pd, y = pb),
    colour = "black",
    linewidth = 1,
    linetype = "dashed"
  ) +
  geom_point(
    aes(x = pd, y = pb, colour = inside_population_1sd_ellipse),
    alpha = 0.6,
    size = .5,
    # shape no fill
    shape = 21
  ) +
  scale_colour_manual(
    values = c("FALSE" = "#d95f02", "TRUE" = "#1b9e77"),
    labels = c("FALSE" = "outside", "TRUE" = "inside")
  ) +
  facet_grid(growth_pattern ~ data_points) +
  labs(
    x = "p_d",
    y = "p_b",
    colour = "Inside 95% confidence region?"
  ) +
  theme_minimal()


## Ratios

posterior_patient_samples %>% 
  ggplot(aes(x = pd/pb)) + 
  geom_density() +
  facet_grid(growth_pattern ~ data_points, scales = "free")

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

