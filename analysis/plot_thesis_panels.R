#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(ggplot2)
  library(grid)
  library(jsonlite)
  library(patchwork)
  library(purrr)
  library(readr)
  library(scales)
  library(tidyr)
})

species <- c("MYC", "CDK4", "PDGFRA")
states <- c("NPC-like", "OPC-like", "AC-like", "MES-like")
state_tokens <- c("NPC_like", "OPC_like", "AC_like", "MES_like")
condition_order <- c("ctrl", "P10", "P50", "P250", "R20", "R100", "R500")

# Nature-style contract: quantitative, data-generated panels with restrained
# color, clean axes, editable vector output, and minimal non-data decoration.
palette_contract <- c(
  neutral_dark = "#272727",
  neutral_mid = "#767676",
  neutral_light = "#D8D8D8",
  neutral_pale = "#F2F2F2",
  blue_main = "#0F4D92",
  blue_mid = "#3775BA",
  blue_soft = "#B4C0E4",
  teal = "#42949E",
  green = "#8BCF8B",
  red = "#B64342",
  red_soft = "#F6CFCB",
  orange = "#E28E2C",
  violet = "#6E5E9B"
)

state_colors <- c(
  "NPC-like" = palette_contract[["blue_main"]],
  "OPC-like" = palette_contract[["teal"]],
  "AC-like" = palette_contract[["orange"]],
  "MES-like" = palette_contract[["red"]]
)
species_colors <- c(
  "MYC" = palette_contract[["blue_main"]],
  "CDK4" = palette_contract[["red"]],
  "PDGFRA" = palette_contract[["teal"]]
)
condition_colors <- c(
  "ctrl" = palette_contract[["neutral_dark"]],
  "P10" = palette_contract[["red_soft"]],
  "P50" = "#E39A94",
  "P250" = palette_contract[["red"]],
  "R20" = palette_contract[["blue_soft"]],
  "R100" = palette_contract[["blue_mid"]],
  "R500" = palette_contract[["blue_main"]]
)
condition_shapes <- c(
  "ctrl" = 16,
  "P10" = 21,
  "P50" = 22,
  "P250" = 24,
  "R20" = 21,
  "R100" = 22,
  "R500" = 24
)

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  values <- list(
    input_dir = "results_v4/t87_conditions",
    raw_dir = "raw/t87_drug_bulkfit",
    validation_dir = "outputs/t87_goal_validation",
    output_dir = "outputs/thesis_panels",
    focal_condition = "P10",
    focal_species = "CDK4",
    formats = "pdf,svg"
  )
  for (i in seq_along(args)) {
    if (args[[i]] %in% c("--input-dir", "--raw-dir", "--validation-dir", "--output-dir", "--focal-condition", "--focal-species", "--formats")) {
      key <- sub("^--", "", args[[i]])
      key <- gsub("-", "_", key)
      if (i == length(args)) {
        stop("Missing value for ", args[[i]], call. = FALSE)
      }
      values[[key]] <- args[[i + 1]]
    }
  }
  values
}

opts <- parse_args()
dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)
export_formats <- tolower(trimws(strsplit(opts$formats, ",", fixed = TRUE)[[1]]))

theme_thesis <- function(base_size = 6.8) {
  theme_classic(base_size = base_size, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks.length = unit(1.5, "pt"),
      axis.title = element_text(size = base_size, colour = palette_contract[["neutral_dark"]]),
      axis.text = element_text(size = base_size - 0.4, colour = palette_contract[["neutral_dark"]]),
      strip.background = element_blank(),
      strip.text = element_text(size = base_size - 0.1, face = "bold", colour = palette_contract[["neutral_dark"]]),
      plot.title = element_text(size = base_size + 0.8, face = "bold", hjust = 0, colour = palette_contract[["neutral_dark"]]),
      plot.subtitle = element_text(size = base_size - 0.2, colour = palette_contract[["neutral_mid"]], margin = margin(b = 3)),
      plot.caption = element_text(size = base_size - 1, colour = palette_contract[["neutral_mid"]], hjust = 0),
      legend.title = element_text(size = base_size - 0.2, colour = palette_contract[["neutral_dark"]]),
      legend.text = element_text(size = base_size - 0.6, colour = palette_contract[["neutral_dark"]]),
      legend.key.size = unit(3.5, "mm"),
      legend.background = element_blank(),
      legend.box.background = element_blank(),
      legend.position = "right",
      panel.grid = element_blank(),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA)
    )
}

theme_set(theme_thesis())

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

safe_read_csv <- function(path) {
  if (!file.exists(path)) {
    return(tibble())
  }
  readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
}

safe_read_parquet <- function(path) {
  if (!file.exists(path)) {
    return(tibble())
  }
  arrow::read_parquet(path) |> as_tibble()
}

condition_label <- function(condition) {
  labels <- c(
    "ctrl" = "Control",
    "P10" = "CDK4i 10 nM",
    "P50" = "CDK4i 50 nM",
    "P250" = "CDK4i 250 nM",
    "R20" = "PDGFRAi 20 nM",
    "R100" = "PDGFRAi 100 nM",
    "R500" = "PDGFRAi 500 nM"
  )
  out <- unname(labels[condition])
  ifelse(is.na(out), condition, out)
}

condition_drug <- function(condition) {
  case_when(
    grepl("^P", condition) ~ "CDK4 inhibitor",
    grepl("^R", condition) ~ "PDGFRA inhibitor",
    TRUE ~ "Control"
  )
}

condition_dose <- function(condition) {
  suppressWarnings(as.numeric(gsub("^[A-Za-z]+", "", condition))) |> replace_na(0)
}

target_species_for_condition <- function(condition) {
  case_when(
    grepl("^P", condition) ~ "CDK4",
    grepl("^R", condition) ~ "PDGFRA",
    TRUE ~ "MYC"
  )
}

has_cols <- function(data, columns) {
  all(columns %in% names(data)) && nrow(data) > 0
}

save_panel <- function(plot, filename, width = 7, height = 5) {
  base_path <- file.path(opts$output_dir, tools::file_path_sans_ext(filename))
  pdf_path <- paste0(base_path, ".pdf")

  if ("pdf" %in% export_formats) {
    grDevices::cairo_pdf(pdf_path, width = width, height = height, family = "Arial")
    print(plot)
    grDevices::dev.off()
  }

  if ("svg" %in% export_formats && requireNamespace("svglite", quietly = TRUE)) {
    svglite::svglite(paste0(base_path, ".svg"), width = width, height = height, system_fonts = list(sans = "Arial"))
    print(plot)
    grDevices::dev.off()
  }

  if (any(c("tif", "tiff") %in% export_formats) && requireNamespace("ragg", quietly = TRUE)) {
    ragg::agg_tiff(paste0(base_path, ".tiff"), width = width, height = height, units = "in", res = 600, compression = "lzw")
    print(plot)
    grDevices::dev.off()
  }

  pdf_path
}

record_status <- function(panel_id, title, status, output_file = NA_character_, reason = NA_character_, required_data = NA_character_) {
  tibble(
    panel_id = panel_id,
    title = title,
    status = status,
    output_file = output_file,
    reason = reason,
    required_data = required_data
  )
}

read_condition_tables <- function(input_dir) {
  dirs <- list.dirs(input_dir, full.names = TRUE, recursive = FALSE)
  if (!length(dirs)) {
    return(list())
  }

  names(dirs) <- basename(dirs)
  tables <- map(dirs, function(condition_dir) {
    table_dir <- file.path(condition_dir, "tables")
    list(
      time_summary = safe_read_csv(file.path(table_dir, "time_summary.csv")),
      observations = safe_read_csv(file.path(table_dir, "observations.csv")),
      selected_plot_timepoints = safe_read_csv(file.path(table_dir, "selected_plot_timepoints.csv")),
      cell_snapshots = safe_read_parquet(file.path(table_dir, "cell_snapshots.parquet")),
      events = safe_read_parquet(file.path(table_dir, "events.parquet")),
      lineage_edges = safe_read_parquet(file.path(table_dir, "lineage_edges.parquet")),
      metadata = if (file.exists(file.path(table_dir, "metadata.json"))) fromJSON(file.path(table_dir, "metadata.json")) else list()
    )
  })
  tables[condition_order[condition_order %in% names(tables)]] |>
    append(tables[setdiff(names(tables), condition_order)])
}

bind_table <- function(tables, name) {
  if (!length(tables)) {
    return(tibble())
  }
  imap_dfr(tables, function(x, condition) {
    table <- x[[name]]
    if (!nrow(table)) {
      return(tibble())
    }
    if (!"condition" %in% names(table)) {
      table$condition <- condition
    }
    table
  })
}

sim_tables <- read_condition_tables(opts$input_dir)
time_summary <- bind_table(sim_tables, "time_summary")
observations <- bind_table(sim_tables, "observations")
cell_snapshots <- bind_table(sim_tables, "cell_snapshots")
events <- bind_table(sim_tables, "events")
lineage_edges <- bind_table(sim_tables, "lineage_edges")

raw_ddpcr <- safe_read_csv(file.path(opts$raw_dir, "ddpcr.csv"))
raw_cell_count <- safe_read_csv(file.path(opts$raw_dir, "cell_count.csv"))
raw_flow <- safe_read_csv(file.path(opts$raw_dir, "flow3.csv"))
raw_metadata <- safe_read_csv(file.path(opts$raw_dir, "metadata.csv"))

timeline <- safe_read_csv(file.path(opts$validation_dir, "timeline_comparison.csv"))
day56 <- safe_read_csv(file.path(opts$validation_dir, "day56_summary.csv"))
growth_metrics <- safe_read_csv(file.path(opts$validation_dir, "growth_curve_metrics.csv"))

status <- list()

missing_panel <- function(panel_id, title, required_data) {
  record_status(panel_id, title, "blocked_missing_data", reason = "Required data table or columns were not found.", required_data = required_data)
}

skipped_schematic <- function(panel_id, title) {
  record_status(panel_id, title, "skipped_schematic", reason = "Schematic-only panel skipped per user instruction.", required_data = "Not data-driven.")
}

time_copy_long <- function() {
  if (!has_cols(time_summary, c("condition", "time", paste0("mean_copy_", species)))) {
    return(tibble(condition = character(), time = numeric(), species = character(), mean_copy = numeric(), condition_label = character()))
  }
  time_summary |>
    select(condition, time, all_of(paste0("mean_copy_", species))) |>
    pivot_longer(starts_with("mean_copy_"), names_to = "species", values_to = "mean_copy") |>
    mutate(
      species = sub("^mean_copy_", "", species),
      condition = factor(condition, levels = condition_order),
      condition_label = condition_label(as.character(condition))
    )
}

state_fraction_long <- function(source = c("time_summary", "observations")) {
  source <- match.arg(source)
  if (source == "observations" && has_cols(observations, c("condition", "time", paste0("flow_fraction_", state_tokens)))) {
    data <- observations |>
      select(condition, time, all_of(paste0("flow_fraction_", state_tokens))) |>
      pivot_longer(starts_with("flow_fraction_"), names_to = "state", values_to = "fraction") |>
      mutate(state = sub("^flow_fraction_", "", state))
  } else if (has_cols(time_summary, c("condition", "time", paste0("dominant_fraction_", state_tokens)))) {
    data <- time_summary |>
      select(condition, time, all_of(paste0("dominant_fraction_", state_tokens))) |>
      pivot_longer(starts_with("dominant_fraction_"), names_to = "state", values_to = "fraction") |>
      mutate(state = sub("^dominant_fraction_", "", state))
  } else if (has_cols(time_summary, c("condition", "time", paste0("fraction_", state_tokens)))) {
    data <- time_summary |>
      select(condition, time, all_of(paste0("fraction_", state_tokens))) |>
      pivot_longer(starts_with("fraction_"), names_to = "state", values_to = "fraction") |>
      mutate(state = sub("^fraction_", "", state))
  } else {
    return(tibble(condition = character(), time = numeric(), state = factor(levels = states), fraction = numeric(), condition_label = character()))
  }
  data |>
    mutate(
      state = factor(gsub("_", "-", state, fixed = TRUE), levels = states),
      condition = factor(condition, levels = condition_order),
      condition_label = condition_label(as.character(condition))
    )
}

state_count_long <- function() {
  if (!has_cols(time_summary, c("condition", "time", paste0("dominant_count_", state_tokens)))) {
    return(tibble(condition = character(), time = numeric(), state = factor(levels = states), count = numeric(), condition_label = character()))
  }
  time_summary |>
    select(condition, time, all_of(paste0("dominant_count_", state_tokens))) |>
    pivot_longer(starts_with("dominant_count_"), names_to = "state", values_to = "count") |>
    mutate(
      state = sub("^dominant_count_", "", state),
      state = factor(gsub("_", "-", state, fixed = TRUE), levels = states),
      condition = factor(condition, levels = condition_order),
      condition_label = condition_label(as.character(condition))
    )
}

figure1b_data_matrix <- function() {
  rows <- tibble(
    source = c(
      "Raw ddPCR", "Raw cell count", "Raw flow", "Drug metadata",
      "Simulation time summary", "Cell snapshots", "Event log", "Lineage edges",
      "Observation proxies", "Validation timeline"
    ),
    status = c(
      nrow(raw_ddpcr) > 0,
      nrow(raw_cell_count) > 0,
      nrow(raw_flow) > 0,
      nrow(raw_metadata) > 0,
      nrow(time_summary) > 0,
      nrow(cell_snapshots) > 0,
      nrow(events) > 0,
      nrow(lineage_edges) > 0,
      nrow(observations) > 0,
      nrow(timeline) > 0
    )
  )
  rows
}

panel_figure_1_b <- function() {
  data <- figure1b_data_matrix()
  if (!nrow(data)) {
    return(missing_panel("figure_1_panel_b", "Experimental data availability matrix", "raw/*.csv and exported simulation tables"))
  }
  plot <- data |>
    mutate(
      source = factor(source, levels = rev(source)),
      available = if_else(status, "Available", "Missing")
    ) |>
    ggplot(aes(x = "Data source", y = source, fill = available)) +
    geom_tile(width = 0.8, height = 0.8, color = "white", linewidth = 0.5) +
    geom_text(aes(label = available), size = 2.6, color = palette_contract[["neutral_dark"]]) +
    scale_fill_manual(values = c("Available" = palette_contract[["green"]], "Missing" = palette_contract[["neutral_pale"]])) +
    labs(title = "Experimental and exported data availability", x = NULL, y = NULL, fill = NULL) +
    theme_thesis() +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
  file <- save_panel(plot, "figure_1_panel_b_data_availability_matrix.pdf", 6, 4.6)
  record_status("figure_1_panel_b", "Experimental data availability matrix", "generated", file, required_data = "raw/*.csv and exported simulation tables")
}

panel_figure_1_e <- function() {
  if (nrow(raw_flow) > 0 && has_cols(raw_flow, c("week", "condition", "group", "fraction"))) {
    data <- raw_flow |>
      filter(week == min(week, na.rm = TRUE)) |>
      mutate(
        condition = factor(condition, levels = condition_order),
        group = recode(group, "OLIG2-high" = "NPC/OPC-like", "AC" = "AC-like", "MES" = "MES-like")
      )
    plot <- ggplot(data, aes(x = condition, y = fraction, fill = group)) +
      geom_col(width = 0.75, color = "white", linewidth = 0.25) +
      scale_y_continuous(labels = percent_format(accuracy = 1), expand = expansion(mult = c(0, 0.04))) +
      scale_fill_manual(values = c("NPC/OPC-like" = palette_contract[["blue_main"]], "AC-like" = palette_contract[["orange"]], "MES-like" = palette_contract[["red"]])) +
      labs(title = "Early flow anchor", x = "Condition", y = "Fraction", fill = "Flow group") +
      theme_thesis()
    file <- save_panel(plot, "figure_1_panel_e_early_flow_anchor.pdf", 7, 4.2)
    return(record_status("figure_1_panel_e", "Early flow anchor", "generated", file, required_data = "raw/t87_drug_bulkfit/flow3.csv"))
  }
  if (nrow(state_fraction_long("observations")) > 0) {
    data <- state_fraction_long("observations") |>
      group_by(condition, state) |>
      slice_min(time, n = 1, with_ties = FALSE) |>
      ungroup()
    plot <- ggplot(data, aes(x = condition, y = fraction, fill = state)) +
      geom_col(width = 0.75, color = "white", linewidth = 0.25) +
      scale_y_continuous(labels = percent_format(accuracy = 1), expand = expansion(mult = c(0, 0.04))) +
      scale_fill_manual(values = state_colors) +
      labs(title = "Early simulated flow anchor", x = "Condition", y = "Fraction", fill = "State") +
      theme_thesis()
    file <- save_panel(plot, "figure_1_panel_e_early_flow_anchor.pdf", 7, 4.2)
    return(record_status("figure_1_panel_e", "Early flow anchor", "generated", file, required_data = "observations.csv flow_fraction_* columns"))
  }
  missing_panel("figure_1_panel_e", "Early flow anchor", "raw flow3.csv or observations.csv flow_fraction_* columns")
}

panel_figure_2_a <- function() {
  data <- time_copy_long()
  if (!nrow(data)) {
    return(missing_panel("figure_2_panel_a", "Three-dimensional copy-number space", "time_summary.csv mean_copy_MYC/CDK4/PDGFRA"))
  }
  wide <- data |>
    select(condition, time, species, mean_copy) |>
    pivot_wider(names_from = species, values_from = mean_copy) |>
    mutate(condition = factor(condition, levels = condition_order))
  plot <- wide |>
    pivot_longer(cols = all_of(species), names_to = "y_species", values_to = "y_value") |>
    left_join(
      wide |> pivot_longer(cols = all_of(species), names_to = "x_species", values_to = "x_value"),
      by = c("condition", "time"),
      relationship = "many-to-many"
    ) |>
    filter(as.integer(factor(x_species, levels = species)) < as.integer(factor(y_species, levels = species))) |>
    ggplot(aes(x = x_value, y = y_value, color = condition, group = condition)) +
    geom_path(alpha = 0.45, linewidth = 0.45) +
    geom_point(aes(size = time, shape = condition), fill = "white", alpha = 0.95, stroke = 0.35) +
    facet_grid(y_species ~ x_species, scales = "free") +
    scale_color_manual(values = condition_colors, drop = FALSE, labels = condition_label) +
    scale_shape_manual(values = condition_shapes, guide = "none") +
    scale_size_continuous(range = c(1.2, 3.2)) +
    labs(title = "Multi-ecDNA copy-number space", x = "Mean copy number", y = "Mean copy number", color = "Condition", size = "Time") +
    theme_thesis()
  file <- save_panel(plot, "figure_2_panel_a_copy_number_space.pdf", 8, 6)
  record_status("figure_2_panel_a", "Three-dimensional copy-number space", "generated", file, required_data = "time_summary.csv mean_copy_* columns")
}

pca_scores_from_time_summary <- function() {
  if (!has_cols(time_summary, c("condition", "time", paste0("mean_copy_", species)))) {
    return(list(scores = tibble(), loadings = tibble(), variance = tibble()))
  }
  wide <- time_summary |>
    select(condition, time, all_of(paste0("mean_copy_", species))) |>
    drop_na()
  if (nrow(wide) < 3) {
    return(list(scores = tibble(), loadings = tibble(), variance = tibble()))
  }
  matrix <- wide |> select(all_of(paste0("mean_copy_", species))) |> as.matrix()
  fit <- prcomp(matrix, center = TRUE, scale. = TRUE)
  score_data <- as_tibble(fit$x[, seq_len(min(3, ncol(fit$x))), drop = FALSE]) |>
    bind_cols(wide |> select(condition, time)) |>
    mutate(condition = factor(condition, levels = condition_order))
  loading_data <- as_tibble(fit$rotation[, seq_len(min(3, ncol(fit$rotation))), drop = FALSE], rownames = "feature") |>
    mutate(species = sub("^mean_copy_", "", feature))
  variance_data <- tibble(
    pc = paste0("PC", seq_along(fit$sdev)),
    variance = fit$sdev^2 / sum(fit$sdev^2)
  )
  list(scores = score_data, loadings = loading_data, variance = variance_data)
}

panel_figure_2_b <- function() {
  pca <- pca_scores_from_time_summary()
  if (!has_cols(pca$scores, c("PC1", "PC2", "condition", "time"))) {
    return(missing_panel("figure_2_panel_b", "PCA trajectory plot", "time_summary.csv mean_copy_* columns with at least three rows"))
  }
  var_labels <- pca$variance$variance[1:2] |> percent(accuracy = 1)
  plot <- ggplot(pca$scores, aes(x = PC1, y = PC2, color = condition, group = condition)) +
    geom_path(arrow = arrow(type = "closed", length = unit(0.08, "inches")), linewidth = 0.65, alpha = 0.8) +
    geom_point(aes(size = time, shape = condition), fill = "white", alpha = 0.95, stroke = 0.35) +
    scale_color_manual(values = condition_colors, drop = FALSE, labels = condition_label) +
    scale_shape_manual(values = condition_shapes, guide = "none") +
    scale_size_continuous(range = c(1.5, 3.8)) +
    labs(
      title = "PCA trajectory of simulated ecDNA copy-number phenotype",
      x = paste0("PC1 (", var_labels[[1]], ")"),
      y = paste0("PC2 (", var_labels[[2]], ")"),
      color = "Condition",
      size = "Time"
    ) +
    theme_thesis()
  file <- save_panel(plot, "figure_2_panel_b_pca_trajectory.pdf", 7, 5.4)
  record_status("figure_2_panel_b", "PCA trajectory plot", "generated", file, required_data = "time_summary.csv mean_copy_* columns")
}

panel_figure_2_c <- function() {
  data <- time_copy_long()
  if (!nrow(data)) {
    return(missing_panel("figure_2_panel_c", "Copy-number phenotype clustering", "time_summary.csv mean_copy_* columns"))
  }
  wide <- data |>
    select(condition, time, species, mean_copy) |>
    pivot_wider(names_from = species, values_from = mean_copy) |>
    drop_na()
  if (nrow(wide) < 3) {
    return(missing_panel("figure_2_panel_c", "Copy-number phenotype clustering", "at least three condition-time rows"))
  }
  z <- scale(wide |> select(all_of(species)) |> as.matrix())
  k <- min(4, max(2, nrow(wide) - 1))
  set.seed(1)
  wide$cluster <- factor(kmeans(z, centers = k, nstart = 20)$cluster)
  plot_data <- wide |>
    mutate(sample = paste0(condition, " t", round(time, 2))) |>
    select(sample, condition, time, cluster, all_of(species)) |>
    pivot_longer(all_of(species), names_to = "species", values_to = "mean_copy") |>
    group_by(species) |>
    mutate(z_score = as.numeric(scale(mean_copy))) |>
    ungroup() |>
    arrange(cluster, condition, time) |>
    mutate(sample = factor(sample, levels = unique(sample)))
  plot <- ggplot(plot_data, aes(x = species, y = sample, fill = z_score)) +
    geom_tile(color = "white", linewidth = 0.2) +
    facet_grid(cluster ~ ., scales = "free_y", space = "free_y") +
    scale_fill_gradient2(low = palette_contract[["blue_main"]], mid = "white", high = palette_contract[["red"]], midpoint = 0, name = "z-score") +
    labs(title = "Copy-number phenotype clusters", x = "ecDNA species", y = "Condition-time point") +
    theme_thesis(base_size = 8)
  file <- save_panel(plot, "figure_2_panel_c_copy_number_clustering.pdf", 6, 7)
  record_status("figure_2_panel_c", "Copy-number phenotype clustering", "generated", file, required_data = "time_summary.csv mean_copy_* columns")
}

target_compensation_data <- function() {
  data <- time_copy_long()
  if (!nrow(data)) {
    return(tibble(condition = character(), species = character(), mean_copy = numeric(), ctrl_mean = numeric(), target_species = character(), is_target = logical(), compensation_score = numeric()))
  }
  terminal <- data |>
    group_by(condition, species) |>
    slice_max(time, n = 1, with_ties = FALSE) |>
    ungroup()
  ctrl <- terminal |>
    filter(condition == "ctrl") |>
    select(species, ctrl_mean = mean_copy)
  terminal |>
    left_join(ctrl, by = "species") |>
    mutate(
      target_species = target_species_for_condition(as.character(condition)),
      is_target = species == target_species,
      compensation_score = log2((mean_copy + 1) / (ctrl_mean + 1)),
      condition = factor(condition, levels = condition_order)
    )
}

panel_figure_2_d <- function() {
  data <- target_compensation_data()
  if (!nrow(data)) {
    return(missing_panel("figure_2_panel_d", "Target-copy compensation score", "time_summary.csv mean_copy_* columns with control condition"))
  }
  plot <- ggplot(data, aes(x = condition, y = species, fill = compensation_score)) +
    geom_tile(color = "white", linewidth = 0.4) +
    geom_point(data = filter(data, is_target), shape = 21, fill = NA, color = palette_contract[["neutral_dark"]], size = 2.5, stroke = 0.7) +
    scale_fill_gradient2(low = palette_contract[["blue_main"]], mid = "white", high = palette_contract[["red"]], midpoint = 0, name = "log2 ratio\nvs control") +
    labs(title = "Target-copy compensation score", x = "Condition", y = "ecDNA species") +
    theme_thesis()
  file <- save_panel(plot, "figure_2_panel_d_target_copy_compensation.pdf", 6.4, 4.2)
  record_status("figure_2_panel_d", "Target-copy compensation score", "generated", file, required_data = "time_summary.csv mean_copy_* columns")
}

panel_figure_2_e <- function() {
  data <- target_compensation_data() |>
    filter(condition != "ctrl", is_target) |>
    mutate(rank_label = paste0(condition_label(as.character(condition)), " / ", species))
  if (!nrow(data)) {
    return(missing_panel("figure_2_panel_e", "Select the focal case", "time_summary.csv mean_copy_* columns with treated conditions"))
  }
  plot <- ggplot(data, aes(x = reorder(rank_label, compensation_score), y = compensation_score, fill = as.character(condition) == opts$focal_condition)) +
    geom_col(width = 0.72) +
    coord_flip() +
    scale_fill_manual(values = c("FALSE" = palette_contract[["neutral_light"]], "TRUE" = palette_contract[["red"]]), guide = "none") +
    geom_hline(yintercept = 0, color = palette_contract[["neutral_dark"]], linewidth = 0.3) +
    labs(title = "Ranked target-copy compensation patterns", x = NULL, y = "Target log2 ratio vs control") +
    theme_thesis()
  file <- save_panel(plot, "figure_2_panel_e_focal_case_ranking.pdf", 7, 4.5)
  record_status("figure_2_panel_e", "Select the focal case", "generated", file, required_data = "time_summary.csv mean_copy_* columns")
}

panel_figure_3_c <- function() {
  if (!has_cols(timeline, c("condition", "week", "sim_log10_cell_count", "exp_log10_cell_count"))) {
    return(missing_panel("figure_3_panel_c", "Growth sequential prediction", "outputs/t87_goal_validation/timeline_comparison.csv growth columns"))
  }
  data <- timeline |>
    mutate(condition = factor(condition, levels = condition_order)) |>
    select(condition, condition_label, week, sim_log10_cell_count, exp_log10_cell_count) |>
    pivot_longer(c(sim_log10_cell_count, exp_log10_cell_count), names_to = "series", values_to = "log10_count") |>
    mutate(series = recode(series, sim_log10_cell_count = "Simulated", exp_log10_cell_count = "Observed"))
  plot <- ggplot(data, aes(x = week, y = log10_count, color = series, linetype = series)) +
    geom_line(linewidth = 0.75, na.rm = TRUE) +
    geom_point(size = 1.6, na.rm = TRUE) +
    facet_wrap(~ condition_label, ncol = 4, scales = "free_y") +
    scale_color_manual(values = c("Observed" = palette_contract[["neutral_dark"]], "Simulated" = palette_contract[["blue_main"]])) +
    labs(title = "Growth reconstruction from validation output", x = "Week", y = "log10 cell count", color = NULL, linetype = NULL) +
    theme_thesis(base_size = 8)
  file <- save_panel(plot, "figure_3_panel_c_growth_sequential_prediction.pdf", 9, 5.2)
  record_status("figure_3_panel_c", "Growth sequential prediction", "generated", file, required_data = "timeline_comparison.csv growth columns")
}

panel_figure_3_d <- function() {
  required <- c("condition", "week", paste0("copy_percent_error_", species))
  if (!has_cols(timeline, required)) {
    return(missing_panel("figure_3_panel_d", "ddPCR sequential prediction", "outputs/t87_goal_validation/timeline_comparison.csv copy_percent_error_* columns"))
  }
  data <- timeline |>
    select(condition, week, all_of(paste0("copy_percent_error_", species))) |>
    pivot_longer(starts_with("copy_percent_error_"), names_to = "species", values_to = "percent_error") |>
    mutate(
      species = sub("^copy_percent_error_", "", species),
      condition = factor(condition, levels = condition_order)
    )
  plot <- ggplot(data, aes(x = week, y = condition, fill = percent_error)) +
    geom_tile(color = "white", linewidth = 0.25) +
    facet_wrap(~ species, ncol = 1) +
    scale_fill_gradient2(low = palette_contract[["blue_main"]], mid = "white", high = palette_contract[["red"]], midpoint = 0, name = "% error") +
    labs(title = "ddPCR prediction residuals", x = "Week", y = "Condition") +
    theme_thesis()
  file <- save_panel(plot, "figure_3_panel_d_ddpcr_residual_heatmap.pdf", 7, 6)
  record_status("figure_3_panel_d", "ddPCR sequential prediction", "generated", file, required_data = "timeline_comparison.csv copy_percent_error_* columns")
}

panel_figure_3_e <- function() {
  required <- c("condition", "week", paste0("sim_", species), paste0("exp_", species))
  if (!has_cols(timeline, required)) {
    return(missing_panel("figure_3_panel_e", "Copy-number phenotype reconstruction", "timeline_comparison.csv sim_* and exp_* copy columns"))
  }
  sim <- timeline |> select(condition, week, all_of(paste0("sim_", species))) |> rename_with(~ sub("^sim_", "", .x), starts_with("sim_")) |> mutate(series = "Simulated")
  obs <- timeline |> select(condition, week, all_of(paste0("exp_", species))) |> rename_with(~ sub("^exp_", "", .x), starts_with("exp_")) |> mutate(series = "Observed")
  both <- bind_rows(sim, obs) |> drop_na(all_of(species))
  if (nrow(both) < 4) {
    return(missing_panel("figure_3_panel_e", "Copy-number phenotype reconstruction", "enough non-missing simulated and observed copy rows"))
  }
  fit <- prcomp(both |> select(all_of(species)) |> as.matrix(), center = TRUE, scale. = TRUE)
  scores <- as_tibble(fit$x[, 1:2, drop = FALSE]) |>
    bind_cols(both |> select(condition, week, series)) |>
    mutate(condition = factor(condition, levels = condition_order))
  plot <- ggplot(scores, aes(x = PC1, y = PC2, color = condition, shape = series)) +
    geom_path(aes(group = interaction(condition, series), linetype = series), alpha = 0.55, linewidth = 0.45) +
    geom_point(size = 2.1, alpha = 0.9) +
    scale_color_manual(values = condition_colors, drop = FALSE, labels = condition_label) +
    labs(title = "Observed versus simulated copy-number phenotype space", x = "PC1", y = "PC2", color = "Condition", shape = NULL, linetype = NULL) +
    theme_thesis()
  file <- save_panel(plot, "figure_3_panel_e_pca_reconstruction_overlay.pdf", 7, 5.2)
  record_status("figure_3_panel_e", "Copy-number phenotype reconstruction", "generated", file, required_data = "timeline_comparison.csv sim_* and exp_* copy columns")
}

panel_figure_3_f <- function() {
  status <- missing_panel("figure_3_panel_f", "Baseline comparison", "baseline comparison table with model and metric columns")
  status$reason <- "No baseline model comparison output was found."
  status
}

panel_figure_4_a <- function() {
  required <- c("condition", "sim_log10_cell_count", "exp_log10_cell_count", paste0("sim_", species), paste0("exp_", species))
  if (!has_cols(day56, required)) {
    return(missing_panel("figure_4_panel_a", "Final all-data fit summary", "outputs/t87_goal_validation/day56_summary.csv sim/exp columns"))
  }
  growth <- day56 |>
    transmute(condition, endpoint = "Growth", observed = exp_log10_cell_count, fitted = sim_log10_cell_count)
  copy <- day56 |>
    select(condition, all_of(paste0("sim_", species)), all_of(paste0("exp_", species))) |>
    pivot_longer(-condition, names_to = "name", values_to = "value") |>
    separate(name, c("series", "species"), sep = "_", extra = "merge") |>
    pivot_wider(names_from = series, values_from = value) |>
    transmute(condition, endpoint = species, observed = exp, fitted = sim)
  data <- bind_rows(growth, copy) |> mutate(condition = factor(condition, levels = condition_order))
  plot <- ggplot(data, aes(x = observed, y = fitted, color = condition)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = palette_contract[["neutral_mid"]]) +
    geom_point(size = 2.2, alpha = 0.9) +
    facet_wrap(~ endpoint, scales = "free") +
    scale_color_manual(values = condition_colors, drop = FALSE, labels = condition_label) +
    labs(title = "Final validation summary", x = "Observed", y = "Simulated / fitted", color = "Condition") +
    theme_thesis()
  file <- save_panel(plot, "figure_4_panel_a_final_fit_summary.pdf", 8, 5.2)
  record_status("figure_4_panel_a", "Final all-data fit summary", "generated", file, required_data = "day56_summary.csv sim/exp columns")
}

panel_figure_4_b <- function() {
  data <- state_count_long()
  if (!nrow(data)) {
    return(missing_panel("figure_4_panel_b", "State effective growth heatmap", "time_summary.csv dominant_count_* columns"))
  }
  summary <- data |>
    group_by(condition, state) |>
    arrange(time, .by_group = TRUE) |>
    summarise(
      effective_growth = (log(last(pmax(count, 1))) - log(first(pmax(count, 1)))) / pmax(last(time) - first(time), 1e-9),
      .groups = "drop"
    )
  plot <- ggplot(summary, aes(x = condition, y = state, fill = effective_growth)) +
    geom_tile(color = "white", linewidth = 0.35) +
    scale_fill_gradient2(low = palette_contract[["blue_main"]], mid = "white", high = palette_contract[["red"]], midpoint = 0, name = "log growth\nper time") +
    labs(title = "State effective growth proxy", subtitle = "Derived from exported dominant state counts", x = "Condition", y = "State") +
    theme_thesis()
  file <- save_panel(plot, "figure_4_panel_b_state_effective_growth_heatmap.pdf", 6.8, 4.5)
  record_status("figure_4_panel_b", "State effective growth heatmap", "generated", file, reason = "Uses exported count-growth proxy, not posterior fitted r_eff.", required_data = "time_summary.csv dominant_count_* columns")
}

copy_selection_proxy <- function() {
  data <- time_copy_long()
  if (!nrow(data)) {
    return(tibble(condition = character(), species = character(), beta_proxy = numeric()))
  }
  data |>
    group_by(condition, species) |>
    arrange(time, .by_group = TRUE) |>
    summarise(
      beta_proxy = (log(last(mean_copy) + 1) - log(first(mean_copy) + 1)) / pmax(last(time) - first(time), 1e-9),
      .groups = "drop"
    ) |>
    mutate(condition = factor(condition, levels = condition_order))
}

panel_figure_4_c <- function() {
  data <- copy_selection_proxy()
  if (!nrow(data)) {
    return(missing_panel("figure_4_panel_c", "Copy-selection coefficient heatmap", "time_summary.csv mean_copy_* columns"))
  }
  plot <- ggplot(data, aes(x = condition, y = species, fill = beta_proxy)) +
    geom_tile(color = "white", linewidth = 0.35) +
    scale_fill_gradient2(low = palette_contract[["blue_main"]], mid = "white", high = palette_contract[["red"]], midpoint = 0, name = "copy trend\nproxy") +
    labs(title = "Copy-selection proxy heatmap", subtitle = "Trend in exported mean copy number; not a fitted beta coefficient", x = "Condition", y = "ecDNA species") +
    theme_thesis()
  file <- save_panel(plot, "figure_4_panel_c_copy_selection_proxy_heatmap.pdf", 6.8, 4.3)
  record_status("figure_4_panel_c", "Copy-selection coefficient heatmap", "generated", file, reason = "Uses exported copy trend proxy because fitted beta table is unavailable.", required_data = "time_summary.csv mean_copy_* columns")
}

panel_figure_4_d <- function() {
  data <- target_compensation_data() |>
    filter(condition != "ctrl", is_target) |>
    mutate(drug = condition_drug(as.character(condition)), dose = condition_dose(as.character(condition)))
  if (!nrow(data)) {
    return(missing_panel("figure_4_panel_d", "Partial inhibition versus high-dose cost", "time_summary.csv mean_copy_* columns across dose conditions"))
  }
  plot <- ggplot(data, aes(x = dose, y = compensation_score, color = drug)) +
    geom_hline(yintercept = 0, color = palette_contract[["neutral_mid"]], linetype = "dashed") +
    geom_line(linewidth = 0.75) +
    geom_point(size = 2.4) +
    facet_wrap(~ drug, scales = "free_x") +
    scale_x_continuous(trans = pseudo_log_trans(base = 10), breaks = unique(data$dose)) +
    scale_color_manual(values = c("CDK4 inhibitor" = palette_contract[["red"]], "PDGFRA inhibitor" = palette_contract[["teal"]])) +
    labs(title = "Dose response of target-copy enrichment", x = "Dose (nM, pseudo-log scale)", y = "Target log2 ratio vs control", color = NULL) +
    theme_thesis()
  file <- save_panel(plot, "figure_4_panel_d_target_copy_dose_response.pdf", 7, 4.4)
  record_status("figure_4_panel_d", "Partial inhibition versus high-dose cost", "generated", file, required_data = "time_summary.csv mean_copy_* columns")
}

panel_figure_4_e <- function() {
  tccs <- target_compensation_data() |>
    filter(is_target, condition != "ctrl") |>
    select(condition, species, compensation_score)
  beta <- copy_selection_proxy() |>
    rename(beta_proxy = beta_proxy)
  data <- left_join(tccs, beta, by = c("condition", "species"))
  if (!has_cols(data, c("compensation_score", "beta_proxy"))) {
    return(missing_panel("figure_4_panel_e", "Observed phenotype versus fitted mechanism", "target compensation and copy trend proxy data"))
  }
  label_layer <- if (requireNamespace("ggrepel", quietly = TRUE)) {
    ggrepel::geom_text_repel(size = 3, max.overlaps = 20, show.legend = FALSE)
  } else {
    geom_text(size = 3, nudge_y = 0.01, check_overlap = TRUE, show.legend = FALSE)
  }
  plot <- ggplot(data, aes(x = compensation_score, y = beta_proxy, color = species, label = condition)) +
    geom_hline(yintercept = 0, color = palette_contract[["neutral_light"]]) +
    geom_vline(xintercept = 0, color = palette_contract[["neutral_light"]]) +
    geom_point(size = 2.6) +
    label_layer +
    scale_color_manual(values = species_colors) +
    labs(title = "Phenotype score versus copy-trend proxy", x = "Target-copy compensation score", y = "Copy trend proxy", color = "Species") +
    theme_thesis()
  file <- save_panel(plot, "figure_4_panel_e_phenotype_vs_mechanism_proxy.pdf", 6.2, 4.8)
  record_status("figure_4_panel_e", "Observed phenotype versus fitted mechanism", "generated", file, reason = "Uses exported copy trend proxy because fitted mechanism table is unavailable.", required_data = "time_summary.csv mean_copy_* columns")
}

lineage_ready <- function() {
  has_cols(cell_snapshots, c("condition", "time", "cell_id", "parent_id", paste0("copy_", species), "dominant_state")) &&
    has_cols(lineage_edges, c("condition", "parent_id", "child_id"))
}

lineage_with_founders <- function(focal_condition = opts$focal_condition) {
  if (!lineage_ready()) {
    return(tibble())
  }
  snapshots <- cell_snapshots |> filter(condition == focal_condition)
  edges <- lineage_edges |> filter(condition == focal_condition) |> select(parent_id, child_id)
  if (!nrow(snapshots) || !nrow(edges)) {
    return(tibble())
  }
  parent_by_child <- setNames(as.character(edges$parent_id), as.character(edges$child_id))
  find_founder <- function(cell_id) {
    current <- as.character(cell_id)
    seen <- character()
    while ((current %in% names(parent_by_child)) && !(current %in% seen)) {
      seen <- c(seen, current)
      parent <- parent_by_child[[current]]
      if (is.na(parent)) {
        break
      }
      current <- parent
    }
    current
  }
  snapshots |>
    mutate(founder_id = vapply(as.character(cell_id), find_founder, character(1)))
}

reservoir_summary <- function() {
  snap <- lineage_with_founders()
  if (!nrow(snap)) {
    return(list(snap = tibble(), top_founders = character(), terminal = tibble(), contribution = tibble()))
  }
  copy_col <- paste0("copy_", opts$focal_species)
  terminal_time <- max(snap$time, na.rm = TRUE)
  terminal <- snap |> filter(time == terminal_time)
  contribution <- terminal |>
    group_by(founder_id) |>
    summarise(contribution = sum(.data[[copy_col]], na.rm = TRUE), terminal_cells = n(), .groups = "drop") |>
    arrange(desc(contribution))
  cutoff <- quantile(contribution$contribution, probs = 0.8, na.rm = TRUE)
  top_founders <- contribution |> filter(contribution >= cutoff, contribution > 0) |> pull(founder_id)
  list(snap = snap, top_founders = top_founders, terminal = terminal, contribution = contribution)
}

panel_figure_5_a <- function() {
  pca <- pca_scores_from_time_summary()
  copy_data <- time_copy_long() |> filter(condition == opts$focal_condition, species == opts$focal_species)
  growth_data <- if (has_cols(timeline, c("condition", "week", "sim_log10_cell_count", "exp_log10_cell_count"))) {
    timeline |> filter(condition == opts$focal_condition)
  } else {
    tibble()
  }
  if (!nrow(copy_data) && !has_cols(pca$scores, c("PC1", "PC2"))) {
    return(missing_panel("figure_5_panel_a", "Focal pattern from model-independent analysis", "time_summary.csv mean_copy_* or timeline_comparison.csv"))
  }
  p1 <- if (has_cols(pca$scores, c("PC1", "PC2", "condition", "time"))) {
    ggplot(pca$scores, aes(PC1, PC2, color = condition, group = condition)) +
      geom_path(alpha = 0.4) +
      geom_point(aes(size = time), alpha = 0.8) +
      geom_path(data = filter(pca$scores, condition == opts$focal_condition), color = palette_contract[["red"]], linewidth = 1.1) +
      geom_point(data = filter(pca$scores, condition == opts$focal_condition), color = palette_contract[["red"]], size = 2.4) +
      scale_color_manual(values = condition_colors, drop = FALSE) +
      labs(title = "Focal PCA trajectory", x = "PC1", y = "PC2", color = "Condition", size = "Time") +
      theme_thesis(base_size = 8)
  } else {
    ggplot() + theme_void() + labs(title = "PCA data unavailable")
  }
  p2 <- ggplot(copy_data, aes(time, mean_copy)) +
    geom_line(color = palette_contract[["red"]], linewidth = 0.9) +
    geom_point(color = palette_contract[["red"]], size = 2) +
    labs(title = paste(opts$focal_species, "copy-number trajectory"), x = "Time", y = "Mean copy number") +
    theme_thesis(base_size = 8)
  if (nrow(growth_data)) {
    p2 <- p2 / ggplot(growth_data, aes(week)) +
      geom_line(aes(y = sim_log10_cell_count, color = "Simulated"), linewidth = 0.75) +
      geom_point(aes(y = exp_log10_cell_count, color = "Observed"), size = 1.6, na.rm = TRUE) +
      scale_color_manual(values = c("Observed" = palette_contract[["neutral_dark"]], "Simulated" = palette_contract[["blue_main"]])) +
      labs(title = "Growth validation", x = "Week", y = "log10 cell count", color = NULL) +
      theme_thesis(base_size = 8)
  }
  plot <- p1 | p2
  file <- save_panel(plot, "figure_5_panel_a_focal_pattern.pdf", 9, 4.8)
  record_status("figure_5_panel_a", "Focal pattern from model-independent analysis", "generated", file, required_data = "time_summary.csv and optional timeline_comparison.csv")
}

panel_figure_5_b <- function() {
  res <- reservoir_summary()
  if (!nrow(res$contribution)) {
    return(missing_panel("figure_5_panel_b", "Late contribution concentration", "cell_snapshots.parquet and lineage_edges.parquet for focal condition"))
  }
  data <- res$contribution |>
    mutate(rank = row_number(), cumulative_founders = rank / n(), cumulative_contribution = cumsum(contribution) / sum(contribution))
  plot <- ggplot(data, aes(cumulative_founders, cumulative_contribution)) +
    geom_abline(slope = 1, intercept = 0, color = palette_contract[["neutral_mid"]], linetype = "dashed") +
    geom_line(color = palette_contract[["red"]], linewidth = 1) +
    geom_point(color = palette_contract[["red"]], size = 1.5) +
    scale_x_continuous(labels = percent_format()) +
    scale_y_continuous(labels = percent_format()) +
    labs(title = "Late contribution concentration", x = "Cumulative founder families", y = paste("Cumulative late", opts$focal_species, "contribution")) +
    theme_thesis()
  file <- save_panel(plot, "figure_5_panel_b_late_contribution_lorenz.pdf", 5.6, 4.6)
  record_status("figure_5_panel_b", "Late contribution concentration", "generated", file, required_data = "cell_snapshots.parquet and lineage_edges.parquet")
}

panel_figure_5_c <- function() {
  res <- reservoir_summary()
  if (!nrow(res$snap) || !nrow(res$contribution)) {
    return(missing_panel("figure_5_panel_c", "Ancestral contribution map", "cell_snapshots.parquet and lineage_edges.parquet for focal condition"))
  }
  early <- res$snap |>
    group_by(founder_id) |>
    slice_min(time, n = 1, with_ties = FALSE) |>
    ungroup() |>
    transmute(
      founder_id,
      early_state = dominant_state,
      early_copy = .data[[paste0("copy_", opts$focal_species)]]
    )
  data <- res$contribution |>
    left_join(early, by = "founder_id") |>
    filter(!is.na(early_state)) |>
    mutate(copy_bin = cut_number(early_copy, n = min(4, n_distinct(early_copy)), labels = FALSE)) |>
    group_by(early_state, copy_bin) |>
    summarise(contribution = sum(contribution), .groups = "drop")
  plot <- ggplot(data, aes(x = factor(copy_bin), y = early_state, fill = contribution)) +
    geom_tile(color = "white", linewidth = 0.35) +
    scale_fill_gradient(low = palette_contract[["neutral_pale"]], high = palette_contract[["red"]], name = "Late contribution") +
    labs(title = "Ancestral contribution map", x = paste("Early", opts$focal_species, "copy bin"), y = "Early dominant state") +
    theme_thesis()
  file <- save_panel(plot, "figure_5_panel_c_ancestral_contribution_map.pdf", 6, 4.4)
  record_status("figure_5_panel_c", "Ancestral contribution map", "generated", file, required_data = "cell_snapshots.parquet and lineage_edges.parquet")
}

panel_figure_5_d <- function() {
  res <- reservoir_summary()
  if (!nrow(res$snap) || !length(res$top_founders)) {
    return(missing_panel("figure_5_panel_d", "Selection window", "cell_snapshots.parquet and lineage_edges.parquet for focal condition"))
  }
  data <- res$snap |>
    mutate(group = if_else(founder_id %in% res$top_founders, "Reservoir-like founders", "Background")) |>
    count(time, group, name = "cells") |>
    pivot_wider(names_from = group, values_from = cells, values_fill = 0) |>
    arrange(time)
  if (!"Reservoir-like founders" %in% names(data)) data[["Reservoir-like founders"]] <- 0
  if (!"Background" %in% names(data)) data[["Background"]] <- 0
  baseline <- data |> slice_min(time, n = 1)
  base_ratio <- log((baseline[["Reservoir-like founders"]] + 0.5) / (baseline[["Background"]] + 0.5))
  data <- data |> mutate(enrichment = log((`Reservoir-like founders` + 0.5) / (`Background` + 0.5)) - base_ratio)
  plot <- ggplot(data, aes(time, enrichment)) +
    geom_hline(yintercept = 0, color = palette_contract[["neutral_mid"]], linetype = "dashed") +
    geom_line(color = palette_contract[["red"]], linewidth = 1) +
    geom_point(color = palette_contract[["red"]], size = 1.8) +
    labs(title = "Selection window for focal contributor families", x = "Time", y = "Log enrichment from baseline") +
    theme_thesis()
  file <- save_panel(plot, "figure_5_panel_d_selection_window.pdf", 6.2, 4.3)
  record_status("figure_5_panel_d", "Selection window", "generated", file, required_data = "cell_snapshots.parquet and lineage_edges.parquet")
}

panel_figure_5_e <- function() {
  res <- reservoir_summary()
  if (!nrow(res$snap) || !length(res$top_founders)) {
    return(missing_panel("figure_5_panel_e", "Dynamic copy-number flux", "cell_snapshots.parquet and lineage_edges.parquet for focal condition"))
  }
  data <- res$snap |>
    mutate(group = if_else(founder_id %in% res$top_founders, "Reservoir-like founders", "Background")) |>
    select(time, group, copy = all_of(paste0("copy_", opts$focal_species)))
  plot <- ggplot(data, aes(x = factor(round(time, 2)), y = copy, fill = group)) +
    geom_boxplot(outlier.size = 0.35, width = 0.7, alpha = 0.8) +
    scale_fill_manual(values = c("Reservoir-like founders" = palette_contract[["red"]], "Background" = palette_contract[["neutral_light"]])) +
    labs(title = "Dynamic copy-number flux", x = "Time", y = paste(opts$focal_species, "copy number"), fill = NULL) +
    theme_thesis(base_size = 8) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  file <- save_panel(plot, "figure_5_panel_e_dynamic_copy_number_flux.pdf", 7, 4.6)
  record_status("figure_5_panel_e", "Dynamic copy-number flux", "generated", file, required_data = "cell_snapshots.parquet and lineage_edges.parquet")
}

panel_figure_5_f <- function() {
  res <- reservoir_summary()
  if (!nrow(res$snap) || !length(res$top_founders)) {
    return(missing_panel("figure_5_panel_f", "State route of focal contributors", "cell_snapshots.parquet and lineage_edges.parquet for focal condition"))
  }
  data <- res$snap |>
    filter(founder_id %in% res$top_founders) |>
    count(time, dominant_state, name = "cells") |>
    group_by(time) |>
    mutate(fraction = cells / sum(cells)) |>
    ungroup()
  plot <- ggplot(data, aes(time, fraction, fill = dominant_state)) +
    geom_area(color = "white", linewidth = 0.15, alpha = 0.9) +
    scale_fill_manual(values = state_colors) +
    scale_y_continuous(labels = percent_format()) +
    labs(title = "State route of focal contributors", x = "Time", y = "Fraction", fill = "Dominant state") +
    theme_thesis()
  file <- save_panel(plot, "figure_5_panel_f_state_route.pdf", 6.2, 4.3)
  record_status("figure_5_panel_f", "State route of focal contributors", "generated", file, required_data = "cell_snapshots.parquet and lineage_edges.parquet")
}

panel_figure_5_g <- function() {
  res <- reservoir_summary()
  if (!nrow(res$snap) || !length(res$top_founders)) {
    return(missing_panel("figure_5_panel_g", "Reconstruct the bulk ddPCR curve", "cell_snapshots.parquet and lineage_edges.parquet for focal condition"))
  }
  data <- res$snap |>
    mutate(group = if_else(founder_id %in% res$top_founders, "Reservoir-like founders", "Background")) |>
    group_by(time, group) |>
    summarise(contribution = sum(.data[[paste0("copy_", opts$focal_species)]], na.rm = TRUE), .groups = "drop")
  plot <- ggplot(data, aes(time, contribution, fill = group)) +
    geom_area(alpha = 0.9, color = "white", linewidth = 0.15) +
    scale_fill_manual(values = c("Reservoir-like founders" = palette_contract[["red"]], "Background" = palette_contract[["neutral_light"]])) +
    labs(title = "Bulk copy-number reconstruction by contributor group", x = "Time", y = paste("Summed", opts$focal_species, "copies"), fill = NULL) +
    theme_thesis()
  file <- save_panel(plot, "figure_5_panel_g_bulk_reconstruction.pdf", 6.2, 4.3)
  record_status("figure_5_panel_g", "Reconstruct the bulk ddPCR curve", "generated", file, required_data = "cell_snapshots.parquet and lineage_edges.parquet")
}

panel_s1 <- function() {
  if (!nrow(raw_ddpcr) && !nrow(raw_cell_count) && !nrow(raw_flow)) {
    return(missing_panel("supplementary_figure_s1", "Raw data QC and normalization", "raw ddPCR, cell_count, or flow tables"))
  }
  matrix <- tibble(
    table = c("ddPCR", "Cell count", "Flow"),
    rows = c(nrow(raw_ddpcr), nrow(raw_cell_count), nrow(raw_flow)),
    missing_cells = c(sum(is.na(raw_ddpcr)), sum(is.na(raw_cell_count)), sum(is.na(raw_flow)))
  ) |>
    pivot_longer(c(rows, missing_cells), names_to = "metric", values_to = "value")
  plot <- ggplot(matrix, aes(metric, table, fill = value)) +
    geom_tile(color = "white", linewidth = 0.35) +
    geom_text(aes(label = value), size = 2.7) +
    scale_fill_gradient(low = palette_contract[["neutral_pale"]], high = palette_contract[["blue_main"]]) +
    labs(title = "Raw data QC summary", x = "Metric", y = "Table", fill = "Value") +
    theme_thesis()
  file <- save_panel(plot, "supplementary_figure_s1_raw_data_qc.pdf", 5.8, 3.8)
  record_status("supplementary_figure_s1", "Raw data QC and normalization", "generated", file, required_data = "raw ddPCR, cell_count, or flow tables")
}

panel_s2 <- function() {
  if (!has_cols(timeline, c("condition", "week", "sim_log10_cell_count", paste0("sim_", species)))) {
    return(missing_panel("supplementary_figure_s2", "All raw longitudinal trajectories", "timeline_comparison.csv or full raw longitudinal ddPCR/cell-count data"))
  }
  copy <- timeline |>
    select(condition, week, all_of(paste0("sim_", species))) |>
    pivot_longer(starts_with("sim_"), names_to = "species", values_to = "mean_copy") |>
    mutate(species = sub("^sim_", "", species))
  p1 <- ggplot(copy, aes(week, mean_copy, color = condition)) +
    geom_line(linewidth = 0.65) +
    facet_wrap(~ species, scales = "free_y") +
    scale_color_manual(values = condition_colors, drop = FALSE) +
    labs(title = "All simulated copy-number trajectories", x = "Week", y = "Mean copy number", color = "Condition") +
    theme_thesis(base_size = 8)
  p2 <- ggplot(timeline, aes(week, sim_log10_cell_count, color = condition)) +
    geom_line(linewidth = 0.65) +
    scale_color_manual(values = condition_colors, drop = FALSE) +
    labs(title = "All simulated growth trajectories", x = "Week", y = "log10 cell count", color = "Condition") +
    theme_thesis(base_size = 8)
  plot <- p1 / p2 + plot_layout(heights = c(2, 1))
  file <- save_panel(plot, "supplementary_figure_s2_all_longitudinal_trajectories.pdf", 8, 7)
  record_status("supplementary_figure_s2", "All raw longitudinal trajectories", "generated", file, required_data = "timeline_comparison.csv")
}

panel_s3 <- function() {
  pca <- pca_scores_from_time_summary()
  if (!nrow(pca$variance) || !nrow(pca$loadings)) {
    return(missing_panel("supplementary_figure_s3", "PCA and clustering robustness", "time_summary.csv mean_copy_* columns"))
  }
  p1 <- pca$variance |>
    slice_head(n = 3) |>
    ggplot(aes(pc, variance)) +
    geom_col(fill = palette_contract[["blue_main"]], width = 0.65) +
    scale_y_continuous(labels = percent_format()) +
    labs(title = "PCA explained variance", x = NULL, y = "Variance explained") +
    theme_thesis(base_size = 8)
  p2 <- pca$loadings |>
    pivot_longer(starts_with("PC"), names_to = "pc", values_to = "loading") |>
    ggplot(aes(species, loading, fill = species)) +
    geom_col(width = 0.65) +
    facet_wrap(~ pc) +
    scale_fill_manual(values = species_colors, guide = "none") +
    labs(title = "PCA loadings", x = NULL, y = "Loading") +
    theme_thesis(base_size = 8)
  plot <- p1 | p2
  file <- save_panel(plot, "supplementary_figure_s3_pca_robustness.pdf", 8, 4)
  record_status("supplementary_figure_s3", "PCA and clustering robustness", "generated", file, required_data = "time_summary.csv mean_copy_* columns")
}

panel_s4 <- function() {
  if (!has_cols(timeline, c("condition", "week", "log10_cell_error", paste0("copy_percent_error_", species)))) {
    return(missing_panel("supplementary_figure_s4", "Sequential validation details", "timeline_comparison.csv residual columns"))
  }
  copy <- timeline |>
    select(condition, week, all_of(paste0("copy_percent_error_", species))) |>
    pivot_longer(starts_with("copy_percent_error_"), names_to = "metric", values_to = "error") |>
    mutate(metric = sub("^copy_percent_error_", "", metric))
  growth <- timeline |> transmute(condition, week, metric = "Growth log10", error = log10_cell_error)
  data <- bind_rows(copy, growth)
  plot <- ggplot(data, aes(week, error, color = condition)) +
    geom_hline(yintercept = 0, color = palette_contract[["neutral_mid"]], linetype = "dashed") +
    geom_line(linewidth = 0.6, na.rm = TRUE) +
    facet_wrap(~ metric, scales = "free_y") +
    scale_color_manual(values = condition_colors, drop = FALSE) +
    labs(title = "Sequential validation residual details", x = "Week", y = "Residual", color = "Condition") +
    theme_thesis(base_size = 8)
  file <- save_panel(plot, "supplementary_figure_s4_sequential_validation_details.pdf", 8, 5)
  record_status("supplementary_figure_s4", "Sequential validation details", "generated", file, required_data = "timeline_comparison.csv residual columns")
}

panel_s11 <- function() {
  if (!has_cols(cell_snapshots, c("condition", "time", "dominant_state", paste0("copy_", species)))) {
    return(missing_panel("supplementary_figure_s11", "Exploratory model-inferred ecDNA-state association scores", "cell_snapshots.parquet dominant_state and copy_* columns"))
  }
  entropy <- function(p) {
    p <- p[p > 0]
    -sum(p * log(p))
  }
  mutual_info <- function(state, copy) {
    if (length(unique(state)) < 2 || length(unique(copy)) < 2) {
      return(0)
    }
    bins <- cut_number(copy, n = min(4, length(unique(copy))), labels = FALSE)
    joint <- table(state, bins)
    total <- sum(joint)
    pxy <- joint / total
    px <- rowSums(pxy)
    py <- colSums(pxy)
    value <- 0
    for (i in seq_len(nrow(pxy))) {
      for (j in seq_len(ncol(pxy))) {
        if (pxy[i, j] > 0 && px[i] > 0 && py[j] > 0) {
          value <- value + pxy[i, j] * log(pxy[i, j] / (px[i] * py[j]))
        }
      }
    }
    h <- entropy(as.numeric(table(state)) / length(state))
    if (h <= 0) 0 else value / h
  }
  data <- crossing(
    condition = unique(cell_snapshots$condition),
    time = unique(cell_snapshots$time),
    species = species
  ) |>
    mutate(score = pmap_dbl(list(condition, time, species), function(condition, time, species) {
      subset <- cell_snapshots |> filter(.data$condition == condition, .data$time == time)
      if (!nrow(subset)) return(NA_real_)
      mutual_info(subset$dominant_state, subset[[paste0("copy_", species)]])
    })) |>
    filter(!is.na(score)) |>
    mutate(condition = factor(condition, levels = condition_order))
  if (!nrow(data)) {
    return(missing_panel("supplementary_figure_s11", "Exploratory model-inferred ecDNA-state association scores", "non-empty cell snapshots by condition/time/species"))
  }
  plot <- ggplot(data, aes(x = time, y = condition, fill = score)) +
    geom_tile(color = "white", linewidth = 0.25) +
    facet_wrap(~ species, ncol = 1) +
    scale_fill_gradient(low = palette_contract[["neutral_pale"]], high = palette_contract[["violet"]], name = "mESIS") +
    labs(title = "Exploratory ecDNA-state information score", subtitle = "Model-inferred from exported cell snapshots", x = "Time", y = "Condition") +
    theme_thesis()
  file <- save_panel(plot, "supplementary_figure_s11_exploratory_mesis.pdf", 7, 6)
  record_status("supplementary_figure_s11", "Exploratory model-inferred ecDNA-state association scores", "generated", file, reason = "Exploratory model-inferred score; not experimentally measured.", required_data = "cell_snapshots.parquet")
}

panel_s12 <- function() {
  if (!length(sim_tables)) {
    return(missing_panel("supplementary_figure_s12", "Workflow and reproducibility", "exported condition directories with tables/manifest.csv or metadata.json"))
  }
  data <- imap_dfr(sim_tables, function(x, condition) {
    manifest_path <- file.path(opts$input_dir, condition, "tables", "manifest.csv")
    manifest <- safe_read_csv(manifest_path)
    if (!nrow(manifest)) {
      return(tibble(condition = condition, table = names(x)[map_lgl(x, is.data.frame)], rows = map_int(x[map_lgl(x, is.data.frame)], nrow)))
    }
    manifest |> mutate(condition = condition)
  })
  plot <- data |>
    mutate(condition = factor(condition, levels = condition_order), table = factor(table, levels = rev(unique(table)))) |>
    ggplot(aes(condition, table, fill = rows)) +
    geom_tile(color = "white", linewidth = 0.3) +
    geom_text(aes(label = rows), size = 2.5) +
    scale_fill_gradient(low = palette_contract[["neutral_pale"]], high = palette_contract[["blue_main"]]) +
    labs(title = "Exported table manifest", x = "Condition", y = "Table", fill = "Rows") +
    theme_thesis(base_size = 8)
  file <- save_panel(plot, "supplementary_figure_s12_export_manifest.pdf", 7.5, 5)
  record_status("supplementary_figure_s12", "Workflow and reproducibility", "generated", file, required_data = "manifest.csv or exported condition tables")
}

panels <- list(
  function() skipped_schematic("figure_1_panel_a", "Biological system schematic"),
  panel_figure_1_b,
  function() skipped_schematic("figure_1_panel_c", "Observed versus inferred layers"),
  function() skipped_schematic("figure_1_panel_d", "Thesis logic flow"),
  panel_figure_1_e,
  panel_figure_2_a,
  panel_figure_2_b,
  panel_figure_2_c,
  panel_figure_2_d,
  panel_figure_2_e,
  function() skipped_schematic("figure_3_panel_a", "Simulator-to-observation schematic"),
  function() skipped_schematic("figure_3_panel_b", "Sequential prediction design"),
  panel_figure_3_c,
  panel_figure_3_d,
  panel_figure_3_e,
  panel_figure_3_f,
  panel_figure_4_a,
  panel_figure_4_b,
  panel_figure_4_c,
  panel_figure_4_d,
  panel_figure_4_e,
  panel_figure_5_a,
  panel_figure_5_b,
  panel_figure_5_c,
  panel_figure_5_d,
  panel_figure_5_e,
  panel_figure_5_f,
  panel_figure_5_g,
  function() missing_panel("figure_5_panel_h", "Virtual continuation prediction", "counterfactual continuation output for continue/escalate/washout scenarios"),
  function() skipped_schematic("figure_6_panel_a", "Ablation design"),
  function() missing_panel("figure_6_panel_b", "Overall reconstruction loss", "ablation reconstruction loss table"),
  function() missing_panel("figure_6_panel_c", "Modality-specific failure", "ablation modality-specific loss table"),
  function() missing_panel("figure_6_panel_d", "Representative failure trajectory", "ablation trajectory output"),
  function() missing_panel("figure_6_panel_e", "Mechanism necessity score heatmap", "mechanism necessity score table"),
  function() skipped_schematic("figure_6_panel_f", "Final working model and claim boundary"),
  panel_s1,
  panel_s2,
  panel_s3,
  panel_s4,
  function() missing_panel("supplementary_figure_s5", "Baseline model comparison", "baseline model comparison table"),
  function() missing_panel("supplementary_figure_s6", "Small simulated populations reproduce large-population summaries", "population-size sensitivity outputs"),
  function() missing_panel("supplementary_figure_s7", "Parameter uncertainty and identifiability", "posterior/parameter uncertainty tables"),
  function() missing_panel("supplementary_figure_s8", "Virtual purification and dynamic equilibrium prediction", "virtual purification simulation outputs"),
  function() missing_panel("supplementary_figure_s9", "Focal case robustness", "multi-seed or posterior focal-case robustness outputs"),
  function() missing_panel("supplementary_figure_s10", "Mechanism ablation details", "ablation detail outputs"),
  panel_s11,
  panel_s12
)

status <- map_dfr(panels, function(panel_fn) {
  tryCatch(
    panel_fn(),
    error = function(e) {
      record_status("unknown", "Panel generation error", "error", reason = conditionMessage(e), required_data = NA_character_)
    }
  )
})

status_path <- file.path(opts$output_dir, "panel_status.csv")
readr::write_csv(status, status_path)

message("Panel output directory: ", normalizePath(opts$output_dir, winslash = "/", mustWork = FALSE))
message("Panel status: ", normalizePath(status_path, winslash = "/", mustWork = FALSE))
message("Generated data-driven panels: ", sum(status$status == "generated", na.rm = TRUE))
message("Export formats: ", paste(export_formats, collapse = ", "))
