# ============================================================

# Print warnings immediately so model-specific warnings are traceable.
options(warn = 1)
# 16S Water + Sediment + Soil integrated 9-model ML workflow
# One script, three habitat-specific models, combined publication figures
#
# IMPORTANT DESIGN:
# - Water, Sediment and Soil are NOT pooled into one prediction model.
# - Each habitat is fitted independently with the same nested-CV framework.
# - Protected main effects: EC + MAP + DL_process + DL_LCBD.
# - DR_process and DR_LCBD are downstream-only variables.
# - Fold-wise Spearman screening removes redundant optional environmental predictors at |rho| >= 0.70.
# - Model-specific RFE is performed within the outer-training data after redundancy screening.
# - Environmental screening and RFE stability are exported across outer folds.
# - Virtual screening uses scenario-specific observed environmental backgrounds:
#   only DL_LCBD and DL_process are overwritten; all other retained predictors
#   remain at each low/high-MAP sample's observed values.
# - Combined radar, OOF-fit, SHAP and virtual-screening figures are exported.
# ============================================================

# -----------------------------
# 0. Input files and output directory
# -----------------------------
INPUT_FILES <- c(
  Water = "d:/Dpan/R data/R data/Tibet_rivers_2023/tibet2023_buchongdata/ml_input_data/16s/Water_16S_ML_master_input_data_20260928.xlsx",
  Sediment = "d:/Dpan/R data/R data/Tibet_rivers_2023/tibet2023_buchongdata/ml_input_data/16s/Sediment_16S_ML_master_input_data_20260928.xlsx",
  Soil = "d:/Dpan/R data/R data/Tibet_rivers_2023/tibet2023_buchongdata/ml_input_data/16s/Soil_16S_ML_master_input_data_20260928.xlsx"
)

MAIN_OUT_DIR <- "d:/Dpan/R data/R data/Tibet_rivers_2023/tibet2023_buchongdata/ml_input_data/16s/16S_3Habitat_9ML_ENV_RHO070_RFE_STABILITY_FINAL1002"
dir.create(MAIN_OUT_DIR, showWarnings = FALSE, recursive = TRUE)

required_extra_packages <- c("patchwork", "cowplot")
missing_extra_packages <- required_extra_packages[
  !vapply(required_extra_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_extra_packages) > 0) {
  stop("Please install the following packages first: ",
       paste(missing_extra_packages, collapse = ", "))
}

# ============================================================
# Global figure colors
# These objects must remain outside run_habitat() because the combined
# cross-habitat figures are generated after run_habitat() returns.
# ============================================================
HABITAT_COLORS_GLOBAL <- c(
  "Water" = "#1F78B4",
  "Sediment" = "#B15928",
  "Soil" = "#8491B4FF"
)

FIT_LINE_COLORS_GLOBAL <- c(
  "Water" = "#145A86",
  "Sediment" = "#7A3D16",
  "Soil" = "#59657D"
)

run_habitat <- function(HABITAT, INPUT_FILE, MAIN_OUT_DIR) {
  # 16S habitat-specific 9-model nested machine-learning workflow
  # Final design: protected core definition -> fold-wise Spearman environmental
  # redundancy screening -> model-specific RFE -> nested tuning/evaluation ->
  # stability audit -> SHAP -> MAP-tail virtual screening.
  # Longitude, latitude and altitude are assumed to have been removed from the input workbook.
  
  # ============================================================
  # 16S habitat-specific machine-learning workflow - TUNED FINAL
  # Input is supplied by INPUT_FILES for the current habitat.
  # Response: Overall_LCBD
  # Protected main effects: EC + MAP + DL_process + DL_LCBD
  # Auxiliary downstream variables: DR_process + DR_LCBD
  #
  # Workflow:
  # 1) Validate habitat input and create reproducible fixed nested CV folds
  # 2) Outer-training Spearman environmental redundancy screening at |rho| >= 0.70
  # 3) Nine separate model-specific inner-CV RFE searches
  # 4) Select each model's minimum inner-RMSE feature set
  # 5) Nine-model nested CV comparison with expanded hyperparameter tuning
  # 6) Radar plot + best-model OOF observed vs predicted
  # 7) DR_process / DR_LCBD downstream linear fits against SAME best-model OOF prediction
  # 8) Full-data refit of the winning model
  # 9) SHAP analysis
  # 10) Virtual screening: DL_process x DL_LCBD averaged over observed bottom/top MAP deciles
  #
  # IMPORTANT:
  # DR_process and DR_LCBD are auxiliary downstream variables only.
  # They never enter screening, RFE, VIF, model fitting/tuning,
  # performance evaluation, SHAP, or virtual screening.
  # ============================================================
  
  # -----------------------------
  # 0. Packages and global settings
  # -----------------------------
  required_packages <- c(
    "readxl", "dplyr", "tidyr", "purrr", "ggplot2",
    "lightgbm", "xgboost", "ranger", "e1071", "FNN", "nnet", "fastshap", "glmnet", "spatialRF"
  )
  
  missing_packages <- required_packages[
    !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
  ]
  
  if (length(missing_packages) > 0) {
    stop(
      "Please install the following packages first: ",
      paste(missing_packages, collapse = ", ")
    )
  }
  
  suppressPackageStartupMessages({
    library(readxl)
    library(dplyr)
    library(tidyr)
    library(purrr)
    library(ggplot2)
    library(lightgbm)
    library(xgboost)
    library(ranger)
    library(e1071)
    library(FNN)
    library(nnet)
    library(fastshap)
  })
  
  set.seed(42)
  
  # Fail fast on deprecated/renamed XGBoost R-interface arguments.
  # This prevents repeated compatibility warnings from being silently propagated
  # through RFE, inner tuning, outer CV, full-data refit, SHAP and screening.
  options("xgboost.strict_mode" = TRUE)
  
  INPUT_SHEET <- "ML_Input"
  OUT_DIR <- file.path(MAIN_OUT_DIR, HABITAT)
  dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
  
  if (!file.exists(INPUT_FILE)) {
    stop("Input file not found for ", HABITAT, ": ", INPUT_FILE)
  }
  
  cat(
    "\n========================================\n",
    "Running habitat: ", HABITAT, "\n",
    "Input file: ", INPUT_FILE, "\n",
    "Output directory: ", OUT_DIR, "\n",
    "========================================\n",
    sep = ""
  )
  
  # Correlation QC is exported after data import and predictor definition.
  
  OUTER_FOLDS <- 5L
  INNER_FOLDS <- 4L
  Y_SCALE <- 1000
  
  habitat_colors <- HABITAT_COLORS_GLOBAL
  fit_line_colors <- FIT_LINE_COLORS_GLOBAL
  
  SCATTER_COLOR <- unname(habitat_colors[HABITAT])
  FIT_LINE_COLOR <- unname(fit_line_colors[HABITAT])
  ONE_TO_ONE_COLOR <- "#4D4D4D"
  
  SHAP_LOW_COLOR <- "#1E88E5"
  SHAP_HIGH_COLOR <- "#FF0051"
  
  VS_COLORS <- c(
    "#2166AC", "#67A9CF", "#D1E5F0",
    "#F7F7F7",
    "#FDDBC7", "#EF8A62", "#B2182B"
  )
  
  MODEL_COLORS_BASE <- c(
    "Elastic Net" = "#9467BD",
    "XGBoost" = "#1F77B4",
    "ANN" = "#FF7F0E",
    "KNN" = "#2CA02C",
    "SVR" = "#D62728",
    "RF" = "#8C564B",
    "LightGBM" = "#E377C2",
    "Ridge" = "#BCBD22",
    "Linear Reg" = "#17BECF"
  )
  
  # -----------------------------
  # 1. Read and validate input
  # -----------------------------
  dat <- read_excel(INPUT_FILE, sheet = INPUT_SHEET) %>%
    as.data.frame(check.names = FALSE)
  
  if (HABITAT == "Water" && !"EC" %in% names(dat)) {
    stop("Water input is missing EC.")
  }
  
  # Harmonize conductivity naming across habitats.
  if (HABITAT == "Sediment") {
    if (!"EC_SD" %in% names(dat)) stop("Sediment input is missing EC_SD.")
    dat$EC <- as.numeric(dat[["EC_SD"]])
  }
  if (HABITAT == "Soil") {
    if (!"conductivity" %in% names(dat)) stop("Soil input is missing conductivity.")
    dat$EC <- as.numeric(dat[["conductivity"]])
  }
  
  required_columns <- c(
    "SampleID", "River", "EC", "MAP",
    "DL_process", "DL_LCBD", "Overall_LCBD",
    "DR_process", "DR_LCBD"
  )
  
  missing_required <- setdiff(required_columns, colnames(dat))
  if (length(missing_required) > 0) {
    stop(
      "Missing required columns: ",
      paste(missing_required, collapse = ", ")
    )
  }
  
  if (nrow(dat) < OUTER_FOLDS * 2) {
    stop("Too few samples for 5-fold nested cross-validation: ", nrow(dat), ".")
  }
  
  if (anyDuplicated(dat$SampleID) > 0) {
    stop("Duplicated SampleID detected.")
  }
  
  if (anyNA(dat[, required_columns, drop = FALSE])) {
    stop("Missing values detected in required columns.")
  }
  
  # Four hypothesis-driven main effects are always retained.
  base_protected_core <- c("EC", "MAP", "DL_process", "DL_LCBD")
  
  protected_core <- base_protected_core
  auxiliary_vars <- c("DR_process", "DR_LCBD")
  response_var <- "Overall_LCBD"
  
  # Coordinates are retained in the master table but intentionally excluded
  # from environmental predictor screening.
  env_candidates <- switch(
    HABITAT,
    "Water" = c(
      "Temp", "pH", "DO", "Na.", "NH4.", "K.", "Mg2.", "Ca2.",
      "F", "CL", "no2", "so4", "no3", "po4",
      "Cropland", "Forest", "Grassland", "Shrubland", "Wetland", "Water",
      "Impervious.surface", "Bareland", "MAT"
    ),
    "Sediment" = c(
      "pH_SD", "TN_SD", "TC_SD", "NH4.N_SD", "NO3.N_SD",
      "Cropland", "Forest", "Grassland", "Shrubland", "Wetland", "Water",
      "Impervious.surface", "Bareland", "MAT"
    ),
    "Soil" = c(
      "pH", "Temp", "TN", "TC", "NH4.N", "NO3.N",
      "Cropland", "Forest", "Grassland", "Shrubland", "Wetland", "Water",
      "Impervious.surface", "Bareland", "MAT"
    ),
    stop("Unknown habitat: ", HABITAT)
  )
  env_candidates <- intersect(env_candidates, colnames(dat))
  
  if (length(intersect(auxiliary_vars, c(protected_core, env_candidates))) > 0) {
    stop("DR_process or DR_LCBD entered the ML predictor pool.")
  }
  
  cat("\nN samples:", nrow(dat), "\n")
  cat("Protected main effects:", paste(base_protected_core, collapse = ", "), "\n")
  cat("Auxiliary variables excluded from ML:", paste(auxiliary_vars, collapse = ", "), "\n")
  cat("Environmental candidates:", paste(env_candidates, collapse = ", "), "\n")
  
  # -----------------------------
  # 1B. Reproducible nested-CV folds
  # -----------------------------
  # Folds are generated independently for each habitat because Water,
  # Sediment and Soil can have different sample sets and sample sizes.
  # The same seed makes the assignments fully reproducible.
  
  make_balanced_folds <- function(n, k, seed) {
    set.seed(seed)
    sample(rep(seq_len(k), length.out = n))
  }
  
  outer_fold_id <- make_balanced_folds(
    n = nrow(dat),
    k = OUTER_FOLDS,
    seed = 42001
  )
  
  outer_folds <- lapply(
    seq_len(OUTER_FOLDS),
    function(f) which(outer_fold_id == f)
  )
  
  inner_fold_by_outer <- setNames(
    vector("list", OUTER_FOLDS),
    as.character(seq_len(OUTER_FOLDS))
  )
  
  for (f in seq_len(OUTER_FOLDS)) {
    tr <- which(outer_fold_id != f)
    inner_ids <- rep(NA_integer_, nrow(dat))
    inner_ids[tr] <- make_balanced_folds(
      n = length(tr),
      k = INNER_FOLDS,
      seed = 43000 + f
    )
    inner_fold_by_outer[[as.character(f)]] <- inner_ids
  }
  
  write.csv(
    data.frame(
      SampleID = dat$SampleID,
      River = dat$River,
      Outer_fold = outer_fold_id
    ),
    file.path(OUT_DIR, "00_outer_CV_fold_assignment.csv"),
    row.names = FALSE
  )
  
  inner_fold_export <- bind_rows(
    lapply(seq_len(OUTER_FOLDS), function(f) {
      data.frame(
        SampleID = dat$SampleID,
        Outer_fold = f,
        Inner_fold = inner_fold_by_outer[[as.character(f)]]
      )
    })
  )
  
  write.csv(
    inner_fold_export,
    file.path(OUT_DIR, "00_inner_CV_fold_assignment.csv"),
    row.names = FALSE
  )
  
  
  # -----------------------------
  # 1C. Core metric helpers
  # -----------------------------
  calc_r2 <- function(obs, pred) {
    denom <- sum((obs - mean(obs))^2)
    if (!is.finite(denom) || denom <= 0) return(NA_real_)
    1 - sum((obs - pred)^2) / denom
  }
  
  calc_metrics <- function(obs, pred) {
    rmse <- sqrt(mean((obs - pred)^2))
    mae <- mean(abs(obs - pred))
    yrange <- diff(range(obs))
    if (!is.finite(yrange) || yrange <= 0) yrange <- 1
    
    data.frame(
      R2 = calc_r2(obs, pred),
      RMSE = rmse,
      MAE = mae,
      Pearson_r = suppressWarnings(cor(obs, pred, method = "pearson")),
      Explained_variance = 1 - var(obs - pred) / var(obs),
      nRMSE = rmse / yrange,
      nMAE = mae / yrange,
      `1-nRMSE` = 1 - rmse / yrange,
      `1-nMAE` = 1 - mae / yrange,
      check.names = FALSE
    )
  }
  
  rmse_vec <- function(obs, pred) {
    sqrt(mean((obs - pred)^2))
  }
  
  # Learn Z-score parameters from the CURRENT training subset only.
  # A constant column is mapped to zero using sd = 1; outer-fold screening
  # separately checks and removes constant optional variables.
  fit_scaler <- function(X) {
    X <- as.matrix(X)
    if (!ncol(X) || any(!is.finite(X))) stop("Invalid training predictors for Z-score")
    mu <- colMeans(X)
    sd0 <- apply(X, 2, sd)
    sd0[!is.finite(sd0) | sd0 == 0] <- 1
    list(mean = mu, sd = sd0)
  }
  
  apply_scaler <- function(X, sc) {
    X <- as.matrix(X)
    if (!identical(colnames(X), names(sc$mean)))
      stop("Prediction columns differ from the training scaler")
    sweep(sweep(X, 2, sc$mean, "-"), 2, sc$sd, "/")
  }
  
  get_inner_fold_indices <- function(outer_f, outer_train_idx) {
    fold_id <- inner_fold_by_outer[[as.character(outer_f)]][outer_train_idx]
    if (anyNA(fold_id)) {
      stop("Missing inner-fold assignment for outer fold ", outer_f)
    }
    lapply(seq_len(INNER_FOLDS), function(k) which(fold_id == k))
  }
  
  
  # -----------------------------
  # 2. Unified environmental screening (inside each outer training fold)
  # -----------------------------
  # The four core predictors are retained. All nine models use the same
  # training-fold screened predictors; tuning occurs inside outer training.
  scale_fit <- function(x) {
    mu <- colMeans(x); sdv <- apply(x, 2, sd)
    sdv[!is.finite(sdv) | sdv == 0] <- 1
    list(mu = mu, sdv = sdv)
  }
  scale_use <- function(x, s) sweep(sweep(x, 2, s$mu, "-"), 2, s$sdv, "/")
  
  # Every preprocessing operation is fitted exclusively to the current training data.
  fit_predict <- function(model, train, test, vars, pars, seed) {
    x <- as.matrix(train[, vars, drop = FALSE]); z <- as.matrix(test[, vars, drop = FALSE])
    storage.mode(x) <- "double"; storage.mode(z) <- "double"
    y <- train$Overall_LCBD * 1000
    if (model %in% c("Linear Reg", "Ridge", "Elastic Net", "KNN", "SVR", "ANN")) {
      sc <- scale_fit(x); x <- scale_use(x, sc); z <- scale_use(z, sc)
    }
    set.seed(seed)
    p <- switch(model,
                "Linear Reg" = {
                  b <- lm.fit(cbind(1, x), y)$coefficients
                  b[!is.finite(b)] <- 0
                  as.numeric(cbind(1, z) %*% b)
                },
                "Ridge" = as.numeric(predict(glmnet::glmnet(x, y, alpha = 0, lambda = pars$lambda,
                                                            standardize = FALSE), newx = z, s = pars$lambda)),
                "Elastic Net" = {
                  fit <- glmnet::glmnet(x, y, alpha = pars$alpha,
                                        lambda = pars$lambda, standardize = FALSE)
                  as.numeric(predict(fit, newx = z, s = pars$lambda))
                },
                "KNN" = FNN::knn.reg(train = x, test = z, y = y, k = pars$k)$pred,
                "SVR" = {
                  g <- if (pars$gamma == "scale") 1 / ncol(x) else as.numeric(pars$gamma)
                  m <- e1071::svm(x = x, y = y, type = "eps-regression", kernel = "radial",
                                  cost = pars$cost, gamma = g, epsilon = pars$epsilon, scale = FALSE)
                  as.numeric(predict(m, z))
                },
                "ANN" = {
                  ym <- mean(y); ys <- sd(y); if (ys == 0) ys <- 1
                  m <- nnet::nnet(x, (y - ym) / ys, size = pars$size, decay = pars$decay,
                                  linout = TRUE, trace = FALSE, maxit = 500, MaxNWts = 3000)
                  as.numeric(predict(m, z)) * ys + ym
                },
                "RF" = {
                  m <- ranger::ranger(x = as.data.frame(x), y = y, num.trees = 300,
                                      mtry = max(1, min(ncol(x), round(ncol(x) * pars$mtry_frac))),
                                      min.node.size = pars$min_leaf, seed = seed, num.threads = 1)
                  as.numeric(predict(m, data = as.data.frame(z))$predictions)
                },
                "XGBoost" = {
                  dtrain <- xgboost::xgb.DMatrix(
                    data = x,
                    label = y
                  )
                  dtest <- xgboost::xgb.DMatrix(data = z)
                  xgb_params <- list(
                    objective = "reg:squarederror",
                    eval_metric = "rmse",
                    max_depth = as.integer(pars$depth),
                    learning_rate = as.numeric(pars$eta),
                    subsample = 0.9,
                    colsample_bytree = 0.9,
                    tree_method = "hist",
                    nthread = 1L,
                    seed = as.integer(seed),
                    verbosity = 0L
                  )
                  m <- xgboost::xgb.train(
                    params = xgb_params,
                    data = dtrain,
                    nrounds = as.integer(pars$nrounds),
                    verbose = 0
                  )
                  as.numeric(predict(m, dtest))
                },
                "LightGBM" = {
                  ds <- lightgbm::lgb.Dataset(x, label = y)
                  m <- lightgbm::lgb.train(params = list(objective = "regression", metric = "rmse",
                                                         learning_rate = pars$eta, num_leaves = pars$leaves, min_data_in_leaf = 5L,
                                                         seed = seed, num_threads = 1L, verbosity = -1L), data = ds,
                                           nrounds = pars$nrounds, verbose = -1)
                  as.numeric(predict(m, z))
                }, stop("Unknown model"))
    as.numeric(p) / 1000
  }
  
  elastic_alpha_grid <- c(0.25, 0.50, 0.75)
  elastic_lambda_grid <- 10^seq(-3, 2, length.out = 20)
  
  parameter_grid <- list(
    "Linear Reg" = data.frame(dummy = 1L),
    "Ridge" = expand.grid(lambda = c(0.01, 0.1, 1, 10)),
    "Elastic Net" = expand.grid(alpha = elastic_alpha_grid,
                                lambda = elastic_lambda_grid),
    "KNN" = expand.grid(k = c(3L, 5L, 9L)),
    "SVR" = expand.grid(cost = c(0.3, 3, 30), gamma = c("scale", "0.1"),
                        epsilon = c(0.01, 0.1), stringsAsFactors = FALSE),
    "ANN" = expand.grid(size = c(3L, 6L), decay = c(0.01, 0.1)),
    "RF" = expand.grid(mtry_frac = c(0.5, 1), min_leaf = c(3L, 6L)),
    "XGBoost" = expand.grid(depth = c(2L, 3L), eta = c(0.03, 0.1),
                            nrounds = c(100L, 250L)),
    "LightGBM" = expand.grid(leaves = c(5L, 9L), eta = c(0.03, 0.1),
                             nrounds = c(100L, 250L))
  )
  
  
  # Unified, deterministic predictor screen; outer-training data only.
  get_vif <- function(data, vars) {
    xx <- as.matrix(data[, vars, drop = FALSE])
    setNames(vapply(seq_along(vars), function(j) {
      yy <- xx[, j]
      zz <- cbind(1, xx[, -j, drop = FALSE])
      fit <- lm.fit(zz, yy)
      rss <- sum(fit$residuals^2)
      tss <- sum((yy - mean(yy))^2)
      if (!is.finite(tss) || tss <= 0) return(Inf)
      if (rss <= .Machine$double.eps * tss) return(Inf)
      tss / rss
    }, numeric(1)), vars)
  }
  
  
  # ============================================================
  # Predictor correlation QC
  # Correlations are reported for diagnosis only.
  # No predictor is removed because of pairwise correlation.
  # ============================================================
  export_correlation_qc <- function(dat, predictors, prefix, out_dir) {
    
    x <- dat[, predictors, drop = FALSE]
    
    cm_pearson <- suppressWarnings(
      cor(x, method = "pearson", use = "pairwise.complete.obs")
    )
    
    cm_spearman <- suppressWarnings(
      cor(x, method = "spearman", use = "pairwise.complete.obs")
    )
    
    write.csv(
      cm_pearson,
      file.path(out_dir, paste0(prefix, "_predictor_correlation_Pearson.csv")),
      row.names = TRUE
    )
    
    write.csv(
      cm_spearman,
      file.path(out_dir, paste0(prefix, "_predictor_correlation_Spearman.csv")),
      row.names = TRUE
    )
    
    make_pairs <- function(cm, method_name) {
      idx <- which(upper.tri(cm), arr.ind = TRUE)
      out <- data.frame(
        Predictor_1 = rownames(cm)[idx[, 1]],
        Predictor_2 = colnames(cm)[idx[, 2]],
        Correlation = cm[idx],
        Abs_correlation = abs(cm[idx]),
        Method = method_name,
        stringsAsFactors = FALSE
      )
      out <- out[order(-out$Abs_correlation), , drop = FALSE]
      rownames(out) <- NULL
      out
    }
    
    pair_table <- rbind(
      make_pairs(cm_pearson, "Pearson"),
      make_pairs(cm_spearman, "Spearman")
    )
    
    write.csv(
      pair_table,
      file.path(out_dir, paste0(prefix, "_predictor_correlation_pairs_QC.csv")),
      row.names = FALSE
    )
    
    invisible(
      list(
        pearson = cm_pearson,
        spearman = cm_spearman,
        pairs = pair_table
      )
    )
  }
  
  # ============================================================
  # Environmental redundancy screening inside each outer-training fold
  # ============================================================
  # Rules:
  # 1) EC, MAP, DL_process and DL_LCBD are protected and never removed.
  # 2) Optional environmental variables with no usable variation are removed.
  # 3) Optional environmental variables correlated with EC or MAP at
  #    |Spearman rho| >= ENV_COR_THRESHOLD are removed; EC/MAP are retained.
  # 4) Remaining optional environmental variables are screened globally at the
  #    same |Spearman rho| threshold. Correlated variables are clustered and one
  #    unsupervised representative is retained from each cluster.
  # 5) DL_process and DL_LCBD are NOT used for environmental redundancy removal.
  # 6) All screening uses outer-training rows only to prevent test-fold leakage.
  
  ENV_COR_THRESHOLD <- 0.70
  ENV_CORE_FOR_REDUNDANCY <- c("EC", "MAP")
  
  safe_spearman <- function(x, y) {
    z <- suppressWarnings(cor(x, y, method = "spearman", use = "pairwise.complete.obs"))
    if (!is.finite(z)) 0 else z
  }
  
  screen_predictors <- function(train, fold_label) {
    bad_core <- protected_core[
      vapply(train[, protected_core, drop = FALSE], function(v) {
        !is.finite(sd(v, na.rm = TRUE)) || sd(v, na.rm = TRUE) < 1e-12
      }, logical(1))
    ]
    if (length(bad_core) > 0) {
      stop("Protected predictor(s) are constant in ", fold_label, ": ",
           paste(bad_core, collapse = ", "))
    }
    
    log_rows <- list()
    add_log <- function(removed, reason, pair = "", value = NA_real_) {
      log_rows[[length(log_rows) + 1L]] <<- data.frame(
        Fold = fold_label, Removed = removed, Reason = reason,
        Pair = pair, Value = value,
        stringsAsFactors = FALSE
      )
    }
    
    usable <- sort(env_candidates)
    
    # Basic QC: remove optional predictors with effectively no variation.
    if (length(usable)) {
      bad_optional <- usable[vapply(train[, usable, drop = FALSE], function(v) {
        !is.finite(sd(v, na.rm = TRUE)) || sd(v, na.rm = TRUE) < 1e-12
      }, logical(1))]
      if (length(bad_optional)) {
        for (v in bad_optional) add_log(v, "constant_or_near_zero_variance")
        usable <- setdiff(usable, bad_optional)
      }
    }
    
    # Protect EC and MAP. Optional environmental variables carrying highly
    # redundant rank information with either core environmental gradient are
    # removed using the same rho = 0.70 threshold. DL variables are deliberately
    # excluded from this redundancy screen.
    if (length(usable)) {
      remove_core <- character(0)
      for (v in usable) {
        rr <- vapply(ENV_CORE_FOR_REDUNDANCY, function(core) {
          abs(safe_spearman(train[[v]], train[[core]]))
        }, numeric(1))
        if (max(rr) >= ENV_COR_THRESHOLD) {
          core <- names(which.max(rr))[1]
          remove_core <- c(remove_core, v)
          add_log(v, "redundant_with_EC_or_MAP",
                  paste(v, core, sep = " ~ "), max(rr))
        }
      }
      usable <- setdiff(usable, unique(remove_core))
    }
    
    # Global optional-environment redundancy control. No response variable is
    # used here. Correlated variables are clustered using absolute Spearman rho.
    # Within each cluster, retain the variable with the lowest mean absolute
    # correlation to EC, MAP and the optional variables outside that cluster.
    kept <- character(0)
    if (length(usable) == 1L) {
      kept <- usable
    } else if (length(usable) > 1L) {
      cm <- suppressWarnings(cor(train[, usable, drop = FALSE],
                                 method = "spearman",
                                 use = "pairwise.complete.obs"))
      cm[!is.finite(cm)] <- 0
      diag(cm) <- 1
      hc <- hclust(as.dist(1 - abs(cm)), method = "average")
      cl <- cutree(hc, h = 1 - ENV_COR_THRESHOLD)
      
      for (cid in sort(unique(cl))) {
        members <- usable[cl == cid]
        if (length(members) == 1L) {
          kept <- c(kept, members)
          next
        }
        
        reference_pool <- unique(c(ENV_CORE_FOR_REDUNDANCY, setdiff(usable, members)))
        redundancy_score <- vapply(members, function(v) {
          if (!length(reference_pool)) return(0)
          mean(vapply(reference_pool, function(ref) {
            abs(safe_spearman(train[[v]], train[[ref]]))
          }, numeric(1)), na.rm = TRUE)
        }, numeric(1))
        
        representative <- names(which.min(redundancy_score))[1]
        kept <- c(kept, representative)
        
        for (v in setdiff(members, representative)) {
          rho_rep <- abs(safe_spearman(train[[v]], train[[representative]]))
          add_log(v, "optional_environment_correlation_cluster",
                  paste(v, representative, sep = " ~ "), rho_rep)
        }
      }
    }
    
    kept <- sort(unique(kept))
    log_df <- if (length(log_rows)) do.call(rbind, log_rows) else data.frame(
      Fold = character(0), Removed = character(0), Reason = character(0),
      Pair = character(0), Value = numeric(0), Step = integer(0),
      stringsAsFactors = FALSE
    )
    if (nrow(log_df)) log_df$Step <- seq_len(nrow(log_df))
    
    list(vars = c(protected_core, kept), log = log_df)
  }
  
  screen_by_fold <- lapply(seq_along(outer_folds), function(f) {
    tr <- setdiff(seq_len(nrow(dat)), outer_folds[[f]])
    screen_predictors(dat[tr, , drop = FALSE], paste0("outer_", f))
  })
  feature_by_fold <- lapply(seq_along(screen_by_fold), function(f) {
    v <- screen_by_fold[[f]]$vars
    data.frame(Outer_fold = f, Predictors = paste(v, collapse = ";"),
               N_environment = length(v) - length(protected_core))
  })
  write.csv(do.call(rbind, feature_by_fold),
            file.path(OUT_DIR, "01_redundancy_screen_by_outer_fold.csv"), row.names = FALSE)
  
  screen_log_all <- do.call(rbind, lapply(screen_by_fold, `[[`, "log"))
  write.csv(screen_log_all,
            file.path(OUT_DIR, "01_environment_redundancy_screen_log.csv"),
            row.names = FALSE)
  
  screening_frequency <- do.call(rbind, lapply(env_candidates, function(variable) {
    present <- vapply(screen_by_fold, function(z) variable %in% z$vars, logical(1))
    data.frame(
      Predictor = variable,
      Retained_outer_folds = sum(present),
      Total_outer_folds = length(present),
      Screening_retention_frequency = mean(present),
      Retained_fold_IDs = paste(which(present), collapse = ";"),
      stringsAsFactors = FALSE
    )
  }))
  write.csv(screening_frequency,
            file.path(OUT_DIR, "01_environment_screening_stability.csv"),
            row.names = FALSE)
  
  # Model-specific RFE: importance is measured by permuting held-out INNER-fold
  # values. Screening, ranking, and selection use only the OUTER training rows.
  RFE_PILOT <- list(
    "Linear Reg" = list(), "Ridge" = list(lambda = 1),
    "Elastic Net" = list(alpha = 0.5, lambda = 0.1),
    "KNN" = list(k = 5L),
    "SVR" = list(cost = 3, gamma = "scale", epsilon = 0.1),
    "ANN" = list(size = 3L, decay = 0.1),
    "RF" = list(mtry_frac = 0.5, min_leaf = 5L),
    "XGBoost" = list(depth = 2L, eta = 0.03, nrounds = 100L),
    "LightGBM" = list(leaves = 5L, eta = 0.03, nrounds = 100L)
  )
  
  rfe_fit <- function(model, train, vars, seed) {
    par <- RFE_PILOT[[model]]
    x <- as.matrix(train[, vars, drop = FALSE])
    yy <- train[[response_var]] * Y_SCALE
    use_scale <- model %in% c("Linear Reg", "Ridge", "Elastic Net", "KNN", "SVR", "ANN")
    if (use_scale) {
      sc <- fit_scaler(x)
      x <- apply_scaler(x, sc)
    }
    transform_new <- function(newdata) {
      zz <- as.matrix(newdata[, vars, drop = FALSE])
      if (use_scale) zz <- apply_scaler(zz, sc)
      zz
    }
    set.seed(seed)
    if (model == "Linear Reg") {
      bb <- lm.fit(cbind(1, x), yy)$coefficients
      bb[!is.finite(bb)] <- 0
      return(function(newdata) as.numeric(cbind(1, transform_new(newdata)) %*% bb) / Y_SCALE)
    }
    if (model %in% c("Ridge", "Elastic Net")) {
      alpha <- if (model == "Ridge") 0 else par$alpha
      fit <- glmnet::glmnet(x, yy, alpha = alpha, lambda = par$lambda,
                            standardize = FALSE)
      return(function(newdata) as.numeric(predict(fit, newx = transform_new(newdata),
                                                  s = par$lambda)) / Y_SCALE)
    }
    if (model == "KNN")
      return(function(newdata) FNN::knn.reg(train = x, test = transform_new(newdata),
                                            y = yy, k = par$k)$pred / Y_SCALE)
    if (model == "SVR") {
      gg <- if (par$gamma == "scale") 1 / ncol(x) else as.numeric(par$gamma)
      fit <- e1071::svm(x = x, y = yy, type = "eps-regression", kernel = "radial",
                        cost = par$cost, gamma = gg, epsilon = par$epsilon, scale = FALSE)
      return(function(newdata) as.numeric(predict(fit, transform_new(newdata))) / Y_SCALE)
    }
    if (model == "ANN") {
      ym <- mean(yy); ys <- sd(yy); if (!is.finite(ys) || ys == 0) ys <- 1
      fit <- nnet::nnet(x, (yy - ym) / ys, size = par$size, decay = par$decay,
                        linout = TRUE, trace = FALSE, maxit = 500, MaxNWts = 3000)
      return(function(newdata) (as.numeric(predict(fit, transform_new(newdata))) * ys + ym) / Y_SCALE)
    }
    if (model == "RF") {
      fit <- ranger::ranger(x = as.data.frame(x), y = yy, num.trees = 300,
                            mtry = max(1L, min(ncol(x), round(ncol(x) * par$mtry_frac))),
                            min.node.size = par$min_leaf, seed = seed, num.threads = 1)
      return(function(newdata) as.numeric(predict(fit,
                                                  data = as.data.frame(transform_new(newdata)))$predictions) / Y_SCALE)
    }
    if (model == "XGBoost") {
      dtrain <- xgboost::xgb.DMatrix(
        data = x,
        label = yy
      )
      xgb_params <- list(
        objective = "reg:squarederror",
        eval_metric = "rmse",
        max_depth = as.integer(par$depth),
        learning_rate = as.numeric(par$eta),
        subsample = 0.9,
        colsample_bytree = 0.9,
        tree_method = "hist",
        nthread = 1L,
        seed = as.integer(seed),
        verbosity = 0L
      )
      fit <- xgboost::xgb.train(
        params = xgb_params,
        data = dtrain,
        nrounds = as.integer(par$nrounds),
        verbose = 0
      )
      return(function(newdata) {
        dnew <- xgboost::xgb.DMatrix(
          data = transform_new(newdata)
        )
        as.numeric(predict(fit, dnew)) / Y_SCALE
      })
    }
    if (model == "LightGBM") {
      fit <- lightgbm::lgb.train(params = list(objective = "regression",
                                               metric = "rmse", learning_rate = par$eta, num_leaves = par$leaves,
                                               min_data_in_leaf = 5L, seed = seed, num_threads = 1L, verbosity = -1L),
                                 data = lightgbm::lgb.Dataset(x, label = yy),
                                 nrounds = par$nrounds, verbose = -1)
      return(function(newdata) as.numeric(predict(fit, transform_new(newdata))) / Y_SCALE)
    }
    stop("Unknown RFE model: ", model)
  }
  
  model_rfe <- function(model, rows, env_pool, fold_label, inner_id) {
    current <- sort(env_pool)
    trace <- list()
    step <- 1L
    # Stable identifiers: removing a variable must not change the random seed
    # used for fitting or permuting any variable still in the candidate set.
    model_code <- match(model, names(RFE_PILOT))
    fold_code <- match(fold_label,
                       c("full_data_final_refit", paste0("outer_", seq_along(outer_folds))))
    if (is.na(model_code) || is.na(fold_code)) stop("Unknown RFE model or fold")
    repeat {
      vars <- c(protected_core, current)
      actual <- numeric(0); predicted <- numeric(0)
      importance <- setNames(rep(0, length(current)), current)
      for (k in sort(unique(inner_id))) {
        tr <- rows[inner_id != k]; va <- rows[inner_id == k]
        predict_this_fit <- rfe_fit(model, dat[tr, , drop = FALSE], vars,
                                    seed = as.integer(41000 + model_code * 1000 + fold_code * 10 + k))
        base <- predict_this_fit(dat[va, , drop = FALSE])
        obs <- dat[[response_var]][va]
        actual <- c(actual, obs); predicted <- c(predicted, base)
        base_mse <- mean((obs - base)^2)
        for (j in seq_along(current)) {
          changed <- dat[va, , drop = FALSE]
          set.seed(as.integer(72000 + model_code * 10000 +
                                fold_code * 1000 + k * 100 + match(current[j], env_candidates)))
          changed[[current[j]]] <- sample(changed[[current[j]]])
          pp <- predict_this_fit(changed)
          importance[current[j]] <- importance[current[j]] +
            (mean((obs - pp)^2) - base_mse) / length(unique(inner_id))
        }
      }
      err <- sqrt(mean((actual - predicted)^2))
      if (!is.finite(err)) stop("RFE non-finite RMSE: ", model, " ", fold_label)
      remove <- if (length(current)) names(which.min(importance)) else NA_character_
      trace[[step]] <- data.frame(Model = model, Fold = fold_label,
                                  Step = step, N_environment = length(current),
                                  Inner_RMSE = err, Inner_R2 = calc_r2(actual, predicted),
                                  Predictors = paste(vars, collapse = ";"),
                                  Removed_next = remove, stringsAsFactors = FALSE)
      if (!length(current)) break
      current <- setdiff(current, remove)
      step <- step + 1L
    }
    history <- do.call(rbind, trace)
    # Exact minimum: ties choose the smaller set, without a fixed R2 tolerance.
    best <- order(history$Inner_RMSE, history$N_environment)[1]
    list(vars = strsplit(history$Predictors[best], ";", fixed = TRUE)[[1]],
         trace = history, best = history[best, , drop = FALSE])
  }
  
  # Only the X variables of an outer training set enter the common redundancy
  # filter. Then each of the nine models carries out its own response-aware RFE.
  ml_models <- names(RFE_PILOT)
  rfe_by_model <- setNames(vector("list", length(ml_models)), ml_models)
  for (model in ml_models) {
    rfe_by_model[[model]] <- lapply(seq_along(outer_folds), function(f) {
      tr <- setdiff(seq_len(nrow(dat)), outer_folds[[f]])
      inner_ids <- inner_fold_by_outer[[as.character(f)]][tr]
      if (anyNA(inner_ids)) stop("Missing inner fold ID for RFE")
      model_rfe(model, tr, setdiff(screen_by_fold[[f]]$vars, protected_core),
                paste0("outer_", f), inner_ids)
    })
    message("Completed model-specific RFE: ", model)
  }
  write.csv(do.call(rbind, lapply(ml_models, function(model)
    do.call(rbind, lapply(rfe_by_model[[model]], `[[`, "trace")))),
    file.path(OUT_DIR, "01_model_specific_RFE_trajectory.csv"), row.names = FALSE)
  write.csv(do.call(rbind, lapply(ml_models, function(model)
    do.call(rbind, lapply(seq_along(outer_folds), function(f) {
      z <- rfe_by_model[[model]][[f]]
      data.frame(Model = model, Outer_fold = f,
                 Predictors = paste(z$vars, collapse = ";"),
                 N_environment = length(z$vars) - length(protected_core),
                 Inner_RMSE = z$best$Inner_RMSE)
    })))), file.path(OUT_DIR, "01_model_specific_RFE_selected_by_fold.csv"),
    row.names = FALSE)
  select_model_features <- function(model, fold, tr) rfe_by_model[[model]][[fold]]$vars
  
  # -----------------------------
  # 3. Model matrices
  # -----------------------------
  X_all <- NULL # assigned after choosing the winning model
  y_all <- dat[[response_var]]
  y_model <- y_all * Y_SCALE
  
  model_oof <- list()
  tuning_log <- list()
  
  # -----------------------------
  # 4A. Linear regression
  # -----------------------------
  pred_linear <- rep(NA_real_, nrow(dat))
  
  for (f in seq_along(outer_folds)) {
    va <- outer_folds[[f]]
    tr <- setdiff(seq_len(nrow(dat)), va)
    model_features <- select_model_features("Linear Reg", f, tr)
    X_all <- as.matrix(dat[, model_features, drop = FALSE])
    
    sc <- fit_scaler(X_all[tr, , drop = FALSE])
    Xtr <- apply_scaler(X_all[tr, , drop = FALSE], sc)
    Xva <- apply_scaler(X_all[va, , drop = FALSE], sc)
    
    fit <- lm.fit(cbind(Intercept = 1, Xtr), y_model[tr])
    
    beta <- fit$coefficients
    beta[!is.finite(beta)] <- 0
    
    pred_linear[va] <- as.numeric(
      cbind(Intercept = 1, Xva) %*% beta
    ) / Y_SCALE
  }
  
  model_oof[["Linear Reg"]] <- pred_linear
  
  # -----------------------------
  # 4B. Ridge regression
  # -----------------------------
  ridge_grid <- 10^seq(-4, 2, length.out = 35)
  
  fit_ridge <- function(Xtr, ytr, Xte, alpha) {
    sc <- fit_scaler(Xtr)
    A <- apply_scaler(Xtr, sc)
    B <- apply_scaler(Xte, sc)
    
    ybar <- mean(ytr)
    yc <- ytr - ybar
    
    beta <- solve(
      crossprod(A) + alpha * diag(ncol(A)),
      crossprod(A, yc)
    )
    
    as.numeric(ybar + B %*% beta)
  }
  
  pred_ridge <- rep(NA_real_, nrow(dat))
  
  for (f in seq_along(outer_folds)) {
    va <- outer_folds[[f]]
    tr <- setdiff(seq_len(nrow(dat)), va)
    model_features <- select_model_features("Ridge", f, tr)
    X_all <- as.matrix(dat[, model_features, drop = FALSE])
    inner <- get_inner_fold_indices(f, tr)
    
    score <- sapply(ridge_grid, function(a) {
      rr <- sapply(inner, function(iva) {
        itr <- setdiff(seq_along(tr), iva)
        
        pr <- fit_ridge(
          X_all[tr[itr], , drop = FALSE],
          y_model[tr[itr]],
          X_all[tr[iva], , drop = FALSE],
          a
        )
        
        rmse_vec(y_model[tr[iva]], pr)
      })
      
      mean(rr)
    })
    
    best_a <- ridge_grid[which.min(score)]
    
    tuning_log[[paste0("Ridge_", f)]] <- data.frame(
      Outer_fold = f,
      alpha = ridge_grid,
      Inner_RMSE = score,
      Selected = ridge_grid == best_a
    )
    
    pred_ridge[va] <- fit_ridge(
      X_all[tr, , drop = FALSE],
      y_model[tr],
      X_all[va, , drop = FALSE],
      best_a
    ) / Y_SCALE
  }
  
  model_oof[["Ridge"]] <- pred_ridge
  
  # -----------------------------
  # 4C. KNN
  # -----------------------------
  knn_grid <- seq(3, 15, by = 2)
  pred_knn <- rep(NA_real_, nrow(dat))
  
  for (f in seq_along(outer_folds)) {
    va <- outer_folds[[f]]
    tr <- setdiff(seq_len(nrow(dat)), va)
    model_features <- select_model_features("KNN", f, tr)
    X_all <- as.matrix(dat[, model_features, drop = FALSE])
    inner <- get_inner_fold_indices(f, tr)
    
    score <- sapply(knn_grid, function(k) {
      rr <- sapply(inner, function(iva) {
        itr <- setdiff(seq_along(tr), iva)
        
        sc <- fit_scaler(X_all[tr[itr], , drop = FALSE])
        Atr <- apply_scaler(X_all[tr[itr], , drop = FALSE], sc)
        Ava <- apply_scaler(X_all[tr[iva], , drop = FALSE], sc)
        
        pr <- FNN::knn.reg(
          train = Atr,
          test = Ava,
          y = y_model[tr[itr]],
          k = k
        )$pred
        
        rmse_vec(y_model[tr[iva]], pr)
      })
      
      mean(rr)
    })
    
    best_k <- knn_grid[which.min(score)]
    
    tuning_log[[paste0("KNN_", f)]] <- data.frame(
      Outer_fold = f,
      k = knn_grid,
      Inner_RMSE = score,
      Selected = knn_grid == best_k
    )
    
    sc <- fit_scaler(X_all[tr, , drop = FALSE])
    
    pred_knn[va] <- FNN::knn.reg(
      train = apply_scaler(X_all[tr, , drop = FALSE], sc),
      test = apply_scaler(X_all[va, , drop = FALSE], sc),
      y = y_model[tr],
      k = best_k
    )$pred / Y_SCALE
  }
  
  model_oof[["KNN"]] <- pred_knn
  
  # -----------------------------
  # 4D. SVR
  # -----------------------------
  # Expanded but still biologically conservative SVR search space.
  # The response is multiplied by Y_SCALE before model fitting, so the original
  # C/epsilon grid was relatively narrow. The broader grid allows both smoother
  # and more flexible radial-basis fits while tuning remains inside inner CV only.
  svr_grid <- expand.grid(
    C = c(0.1, 0.3, 1, 3, 10, 30, 100),
    gamma_code = c("scale", "0.01", "0.03", "0.1", "0.3"),
    epsilon = c(0.01, 0.03, 0.05, 0.1, 0.2, 0.4),
    stringsAsFactors = FALSE
  )
  
  pred_svr <- rep(NA_real_, nrow(dat))
  
  for (f in seq_along(outer_folds)) {
    va <- outer_folds[[f]]
    tr <- setdiff(seq_len(nrow(dat)), va)
    model_features <- select_model_features("SVR", f, tr)
    X_all <- as.matrix(dat[, model_features, drop = FALSE])
    inner <- get_inner_fold_indices(f, tr)
    
    score <- rep(NA_real_, nrow(svr_grid))
    
    for (g in seq_len(nrow(svr_grid))) {
      rr <- sapply(inner, function(iva) {
        itr <- setdiff(seq_along(tr), iva)
        
        sc <- fit_scaler(X_all[tr[itr], , drop = FALSE])
        Atr <- apply_scaler(X_all[tr[itr], , drop = FALSE], sc)
        Ava <- apply_scaler(X_all[tr[iva], , drop = FALSE], sc)
        
        gam <- if (svr_grid$gamma_code[g] == "scale") {
          1 / ncol(Atr)
        } else {
          as.numeric(svr_grid$gamma_code[g])
        }
        
        fit <- e1071::svm(
          x = Atr,
          y = y_model[tr[itr]],
          type = "eps-regression",
          kernel = "radial",
          cost = svr_grid$C[g],
          gamma = gam,
          epsilon = svr_grid$epsilon[g],
          scale = FALSE
        )
        
        pr <- as.numeric(predict(fit, Ava))
        rmse_vec(y_model[tr[iva]], pr)
      })
      
      score[g] <- mean(rr)
    }
    
    best_g <- which.min(score)
    bp <- svr_grid[best_g, ]
    
    tuning_log[[paste0("SVR_", f)]] <- cbind(
      Outer_fold = f,
      svr_grid,
      Inner_RMSE = score,
      Selected = seq_len(nrow(svr_grid)) == best_g
    )
    
    sc <- fit_scaler(X_all[tr, , drop = FALSE])
    Atr <- apply_scaler(X_all[tr, , drop = FALSE], sc)
    Ava <- apply_scaler(X_all[va, , drop = FALSE], sc)
    
    gam <- if (bp$gamma_code == "scale") {
      1 / ncol(Atr)
    } else {
      as.numeric(bp$gamma_code)
    }
    
    fit <- e1071::svm(
      x = Atr,
      y = y_model[tr],
      type = "eps-regression",
      kernel = "radial",
      cost = bp$C,
      gamma = gam,
      epsilon = bp$epsilon,
      scale = FALSE
    )
    
    pred_svr[va] <- as.numeric(predict(fit, Ava)) / Y_SCALE
  }
  
  model_oof[["SVR"]] <- pred_svr
  
  # -----------------------------
  # 4E. ANN
  # -----------------------------
  # Broaden ANN capacity and regularization. The network remains deliberately
  # small because N = 106; larger networks would be unstable and prone to overfit.
  ann_grid <- expand.grid(
    size = c(2, 3, 5, 8, 12),
    decay = c(0.0001, 0.001, 0.01, 0.1, 0.5),
    stringsAsFactors = FALSE
  )
  
  fit_ann_safe <- function(Xtr, ytr, Xte, size, decay, seed) {
    sc_x <- fit_scaler(Xtr)
    A <- apply_scaler(Xtr, sc_x)
    B <- apply_scaler(Xte, sc_x)
    
    y_mu <- mean(ytr)
    y_sd <- sd(ytr)
    if (!is.finite(y_sd) || y_sd == 0) y_sd <- 1
    yz <- (ytr - y_mu) / y_sd
    
    seeds <- seed + 0:2
    for (ss in seeds) {
      set.seed(ss)
      
      fit <- try(
        nnet::nnet(
          x = A,
          y = yz,
          size = size,
          decay = decay,
          linout = TRUE,
          trace = FALSE,
          maxit = 3000,
          MaxNWts = 10000
        ),
        silent = TRUE
      )
      
      if (!inherits(fit, "try-error")) {
        pz <- try(as.numeric(predict(fit, B)), silent = TRUE)
        
        if (!inherits(pz, "try-error") && all(is.finite(pz))) {
          return(y_mu + y_sd * pz)
        }
      }
    }
    
    rep(mean(ytr), nrow(Xte))
  }
  
  pred_ann <- rep(NA_real_, nrow(dat))
  
  for (f in seq_along(outer_folds)) {
    va <- outer_folds[[f]]
    tr <- setdiff(seq_len(nrow(dat)), va)
    model_features <- select_model_features("ANN", f, tr)
    X_all <- as.matrix(dat[, model_features, drop = FALSE])
    inner <- get_inner_fold_indices(f, tr)
    
    score <- rep(NA_real_, nrow(ann_grid))
    
    for (g in seq_len(nrow(ann_grid))) {
      rr <- sapply(seq_along(inner), function(k) {
        iva <- inner[[k]]
        itr <- setdiff(seq_along(tr), iva)
        
        pr <- fit_ann_safe(
          X_all[tr[itr], , drop = FALSE],
          y_model[tr[itr]],
          X_all[tr[iva], , drop = FALSE],
          size = ann_grid$size[g],
          decay = ann_grid$decay[g],
          seed = 46000 + f * 1000 + g * 10 + k
        )
        
        rmse_vec(y_model[tr[iva]], pr)
      })
      
      score[g] <- mean(rr)
    }
    
    best_g <- which.min(score)
    bp <- ann_grid[best_g, ]
    
    tuning_log[[paste0("ANN_", f)]] <- cbind(
      Outer_fold = f,
      ann_grid,
      Inner_RMSE = score,
      Selected = seq_len(nrow(ann_grid)) == best_g
    )
    
    pred_ann[va] <- fit_ann_safe(
      X_all[tr, , drop = FALSE],
      y_model[tr],
      X_all[va, , drop = FALSE],
      size = bp$size,
      decay = bp$decay,
      seed = 47000 + f
    ) / Y_SCALE
  }
  
  model_oof[["ANN"]] <- pred_ann
  
  # -----------------------------
  # 4F. Random forest
  # -----------------------------
  rf_grid <- expand.grid(
    mtry_fraction = c(0.25, 0.50, 0.75, 1.00),
    min.node.size = c(1L, 3L, 5L, 10L),
    sample.fraction = c(0.75, 1.00),
    stringsAsFactors = FALSE
  )
  
  pred_rf <- rep(NA_real_, nrow(dat))
  
  for (f in seq_along(outer_folds)) {
    va <- outer_folds[[f]]
    tr <- setdiff(seq_len(nrow(dat)), va)
    model_features <- select_model_features("RF", f, tr)
    X_all <- as.matrix(dat[, model_features, drop = FALSE])
    inner <- get_inner_fold_indices(f, tr)
    
    score <- rep(NA_real_, nrow(rf_grid))
    
    for (g in seq_len(nrow(rf_grid))) {
      rr <- sapply(seq_along(inner), function(k) {
        iva <- inner[[k]]
        itr <- setdiff(seq_along(tr), iva)
        
        fit <- ranger::ranger(
          x = as.data.frame(X_all[tr[itr], , drop = FALSE]),
          y = y_model[tr[itr]],
          num.trees = 600,
          mtry = max(1L, min(ncol(X_all),
                             as.integer(round(ncol(X_all) * rf_grid$mtry_fraction[g])))),
          min.node.size = rf_grid$min.node.size[g],
          sample.fraction = rf_grid$sample.fraction[g],
          seed = 48000 + f * 1000 + g * 10 + k,
          num.threads = 1
        )
        
        pr <- predict(
          fit,
          data = as.data.frame(X_all[tr[iva], , drop = FALSE])
        )$predictions
        
        rmse_vec(y_model[tr[iva]], pr)
      })
      
      score[g] <- mean(rr)
    }
    
    best_g <- which.min(score)
    bp <- rf_grid[best_g, ]
    
    tuning_log[[paste0("RF_", f)]] <- cbind(
      Outer_fold = f,
      rf_grid,
      Inner_RMSE = score,
      Selected = seq_len(nrow(rf_grid)) == best_g
    )
    
    fit <- ranger::ranger(
      x = as.data.frame(X_all[tr, , drop = FALSE]),
      y = y_model[tr],
      num.trees = 600,
      mtry = max(1L, min(ncol(X_all),
                         as.integer(round(ncol(X_all) * bp$mtry_fraction)))),
      min.node.size = bp$min.node.size,
      sample.fraction = bp$sample.fraction,
      seed = 49000 + f,
      num.threads = 1
    )
    
    pred_rf[va] <- predict(
      fit,
      data = as.data.frame(X_all[va, , drop = FALSE])
    )$predictions / Y_SCALE
  }
  
  model_oof[["RF"]] <- pred_rf
  
  # -----------------------------
  # 4G. XGBoost
  # -----------------------------
  # Expanded XGBoost tuning. A fixed random subset is used to avoid an
  # impractically large Cartesian grid while preserving reproducibility.
  xgb_grid_full <- expand.grid(
    nrounds = c(300L, 600L, 1000L),
    max_depth = c(1L, 2L, 3L, 4L),
    eta = c(0.01, 0.02, 0.03, 0.05),
    gamma = c(0, 0.1, 0.5),
    min_child_weight = c(1, 3, 5, 10),
    subsample = c(0.75, 0.90, 1.00),
    colsample_bytree = c(0.75, 0.90, 1.00),
    lambda = c(0.1, 1, 5),
    alpha = c(0, 0.05, 0.1),
    stringsAsFactors = FALSE
  )
  
  set.seed(50001)
  xgb_grid <- xgb_grid_full[
    sample(seq_len(nrow(xgb_grid_full)), size = 60L),
    ,
    drop = FALSE
  ]
  
  # Always retain a few conservative reference configurations.
  xgb_reference <- data.frame(
    nrounds = c(300L, 600L, 1000L),
    max_depth = c(2L, 3L, 3L),
    eta = c(0.03, 0.02, 0.01),
    gamma = c(0, 0.1, 0),
    min_child_weight = c(1, 3, 5),
    subsample = c(0.80, 0.90, 1.00),
    colsample_bytree = c(0.80, 0.90, 1.00),
    lambda = c(1, 1, 1),
    alpha = c(0, 0.05, 0.1),
    stringsAsFactors = FALSE
  )
  
  xgb_grid <- unique(rbind(xgb_reference, xgb_grid))
  
  pred_xgb <- rep(NA_real_, nrow(dat))
  
  for (f in seq_along(outer_folds)) {
    va <- outer_folds[[f]]
    tr <- setdiff(seq_len(nrow(dat)), va)
    model_features <- select_model_features("XGBoost", f, tr)
    X_all <- as.matrix(dat[, model_features, drop = FALSE])
    inner <- get_inner_fold_indices(f, tr)
    
    score <- rep(NA_real_, nrow(xgb_grid))
    
    for (g in seq_len(nrow(xgb_grid))) {
      rr <- sapply(seq_along(inner), function(k) {
        iva <- inner[[k]]
        itr <- setdiff(seq_along(tr), iva)
        
        params <- list(
          objective = "reg:squarederror",
          eval_metric = "rmse",
          max_depth = as.integer(xgb_grid$max_depth[g]),
          learning_rate = xgb_grid$eta[g],
          gamma = xgb_grid$gamma[g],
          min_child_weight = xgb_grid$min_child_weight[g],
          subsample = xgb_grid$subsample[g],
          colsample_bytree = xgb_grid$colsample_bytree[g],
          lambda = xgb_grid$lambda[g],
          alpha = xgb_grid$alpha[g],
          tree_method = "hist",
          nthread = 1L,
          seed = as.integer(50000 + f * 1000 + g * 10 + k)
        )
        
        fit <- xgboost::xgb.train(
          params = params,
          data = xgboost::xgb.DMatrix(
            X_all[tr[itr], , drop = FALSE],
            label = y_model[tr[itr]]
          ),
          nrounds = as.integer(xgb_grid$nrounds[g]),
          verbose = 0
        )
        
        pr <- predict(
          fit,
          xgboost::xgb.DMatrix(X_all[tr[iva], , drop = FALSE])
        )
        
        rmse_vec(y_model[tr[iva]], pr)
      })
      
      score[g] <- mean(rr)
    }
    
    best_g <- which.min(score)
    bp <- xgb_grid[best_g, ]
    
    tuning_log[[paste0("XGBoost_", f)]] <- cbind(
      Outer_fold = f,
      xgb_grid,
      Inner_RMSE = score,
      Selected = seq_len(nrow(xgb_grid)) == best_g
    )
    
    params <- list(
      objective = "reg:squarederror",
      eval_metric = "rmse",
      max_depth = as.integer(bp$max_depth),
      learning_rate = bp$eta,
      gamma = bp$gamma,
      min_child_weight = bp$min_child_weight,
      subsample = bp$subsample,
      colsample_bytree = bp$colsample_bytree,
      lambda = bp$lambda,
      alpha = bp$alpha,
      tree_method = "hist",
      nthread = 1L,
      seed = as.integer(51000 + f)
    )
    
    fit <- xgboost::xgb.train(
      params = params,
      data = xgboost::xgb.DMatrix(
        X_all[tr, , drop = FALSE],
        label = y_model[tr]
      ),
      nrounds = as.integer(bp$nrounds),
      verbose = 0
    )
    
    pred_xgb[va] <- predict(
      fit,
      xgboost::xgb.DMatrix(X_all[va, , drop = FALSE])
    ) / Y_SCALE
  }
  
  model_oof[["XGBoost"]] <- pred_xgb
  
  # -----------------------------
  # 4H. LightGBM
  # -----------------------------
  # LightGBM tuning emphasizes shallow, regularized fits for small samples.
  # Candidate sampling is fixed before outer-CV assessment.
  lgb_grid_full <- expand.grid(
    num_leaves = c(2L, 3L, 4L, 5L, 7L, 11L, 15L),
    max_depth = c(2L, 3L, 4L, 5L),
    learning_rate = c(0.005, 0.01, 0.02, 0.04, 0.08),
    min_data_in_leaf = c(3L, 6L, 10L, 16L, 24L),
    feature_fraction = c(0.80, 0.90, 1.00),
    bagging_fraction = c(0.80, 0.90, 1.00),
    lambda_l1 = c(0, 0.05, 0.1),
    lambda_l2 = c(0.1, 1, 5),
    nrounds = c(150L, 350L, 700L, 1100L),
    stringsAsFactors = FALSE
  )
  
  # Remove structurally redundant leaf/depth combinations.
  lgb_grid_full <- lgb_grid_full[
    lgb_grid_full$num_leaves <= 2^lgb_grid_full$max_depth,
    ,
    drop = FALSE
  ]
  
  set.seed(52001)
  shallow <- which(lgb_grid_full$num_leaves <= 7L &
                     lgb_grid_full$min_data_in_leaf >= 6L)
  broader <- setdiff(seq_len(nrow(lgb_grid_full)), shallow)
  lgb_grid <- lgb_grid_full[c(
    sample(shallow, size = 48L),
    sample(broader, size = 24L)
  ), , drop = FALSE]
  
  lgb_reference <- data.frame(
    num_leaves = c(2L, 3L, 4L, 5L, 7L, 11L, 15L, 3L),
    max_depth = c(2L, 2L, 3L, 3L, 3L, 4L, 5L, 3L),
    learning_rate = c(0.04, 0.04, 0.02, 0.03, 0.02, 0.02, 0.01, 0.08),
    min_data_in_leaf = c(16L, 10L, 10L, 8L, 6L, 5L, 3L, 16L),
    feature_fraction = c(1, 1, 0.9, 0.8, 0.9, 0.9, 0.9, 1),
    bagging_fraction = c(1, 1, 1, 0.8, 0.9, 0.9, 0.9, 1),
    lambda_l1 = c(0, 0, 0, 0, 0.05, 0.1, 0.1, 0),
    lambda_l2 = c(1, 1, 1, 1, 1, 1, 1, 5),
    nrounds = c(150L, 350L, 700L, 350L, 700L, 700L, 1100L, 150L),
    stringsAsFactors = FALSE
  )
  
  lgb_grid <- unique(rbind(lgb_reference, lgb_grid))
  
  pred_lgb <- rep(NA_real_, nrow(dat))
  
  for (f in seq_along(outer_folds)) {
    va <- outer_folds[[f]]
    tr <- setdiff(seq_len(nrow(dat)), va)
    model_features <- select_model_features("LightGBM", f, tr)
    X_all <- as.matrix(dat[, model_features, drop = FALSE])
    inner <- get_inner_fold_indices(f, tr)
    
    score <- rep(NA_real_, nrow(lgb_grid))
    
    for (g in seq_len(nrow(lgb_grid))) {
      rr <- sapply(seq_along(inner), function(k) {
        iva <- inner[[k]]
        itr <- setdiff(seq_along(tr), iva)
        
        params <- list(
          objective = "regression",
          metric = "rmse",
          num_leaves = as.integer(lgb_grid$num_leaves[g]),
          max_depth = as.integer(lgb_grid$max_depth[g]),
          learning_rate = lgb_grid$learning_rate[g],
          min_data_in_leaf = as.integer(lgb_grid$min_data_in_leaf[g]),
          feature_fraction = lgb_grid$feature_fraction[g],
          bagging_fraction = lgb_grid$bagging_fraction[g],
          bagging_freq = ifelse(lgb_grid$bagging_fraction[g] < 1, 1L, 0L),
          lambda_l1 = lgb_grid$lambda_l1[g],
          lambda_l2 = lgb_grid$lambda_l2[g],
          seed = as.integer(52000 + f * 1000 + g * 10 + k),
          deterministic = TRUE,
          force_col_wise = TRUE,
          verbosity = -1,
          num_threads = 1L
        )
        
        fit <- lgb.train(
          params = params,
          data = lgb.Dataset(
            X_all[tr[itr], , drop = FALSE],
            label = y_model[tr[itr]]
          ),
          nrounds = as.integer(lgb_grid$nrounds[g]),
          verbose = -1
        )
        
        pr <- predict(
          fit,
          X_all[tr[iva], , drop = FALSE]
        )
        
        rmse_vec(y_model[tr[iva]], pr)
      })
      
      score[g] <- mean(rr)
    }
    
    best_g <- which.min(score)
    bp <- lgb_grid[best_g, ]
    
    tuning_log[[paste0("LightGBM_", f)]] <- cbind(
      Outer_fold = f,
      lgb_grid,
      Inner_RMSE = score,
      Selected = seq_len(nrow(lgb_grid)) == best_g
    )
    
    params <- list(
      objective = "regression",
      metric = "rmse",
      num_leaves = as.integer(bp$num_leaves),
      max_depth = as.integer(bp$max_depth),
      learning_rate = bp$learning_rate,
      min_data_in_leaf = as.integer(bp$min_data_in_leaf),
      feature_fraction = bp$feature_fraction,
      bagging_fraction = bp$bagging_fraction,
      bagging_freq = ifelse(bp$bagging_fraction < 1, 1L, 0L),
      lambda_l1 = bp$lambda_l1,
      lambda_l2 = bp$lambda_l2,
      seed = as.integer(53000 + f),
      deterministic = TRUE,
      force_col_wise = TRUE,
      verbosity = -1,
      num_threads = 1L
    )
    
    fit <- lgb.train(
      params = params,
      data = lgb.Dataset(
        X_all[tr, , drop = FALSE],
        label = y_model[tr]
      ),
      nrounds = as.integer(bp$nrounds),
      verbose = -1
    )
    
    pred_lgb[va] <- predict(
      fit,
      X_all[va, , drop = FALSE]
    ) / Y_SCALE
  }
  
  model_oof[["LightGBM"]] <- pred_lgb
  
  # -----------------------------
  # 4I. Elastic Net: tune alpha and lambda on inner folds only
  # -----------------------------
  elastic_grid <- parameter_grid[["Elastic Net"]]
  
  fit_elastic_net <- function(Xtr, ytr, Xte, alpha, lambda) {
    sc <- fit_scaler(Xtr)
    a <- apply_scaler(Xtr, sc)
    b <- apply_scaler(Xte, sc)
    fit <- glmnet::glmnet(a, ytr, alpha = alpha, lambda = lambda,
                          standardize = FALSE)
    as.numeric(predict(fit, newx = b, s = lambda))
  }
  
  pred_elastic <- rep(NA_real_, nrow(dat))
  for (f in seq_along(outer_folds)) {
    va <- outer_folds[[f]]
    tr <- setdiff(seq_len(nrow(dat)), va)
    vars <- select_model_features("Elastic Net", f, tr)
    XX <- as.matrix(dat[, vars, drop = FALSE])
    inner <- get_inner_fold_indices(f, tr)
    inner_error <- matrix(NA_real_, nrow(elastic_grid), length(inner))
    for (k in seq_along(inner)) {
      iv <- inner[[k]]
      it <- setdiff(seq_along(tr), iv)
      sc <- fit_scaler(XX[tr[it], , drop = FALSE])
      a <- apply_scaler(XX[tr[it], , drop = FALSE], sc)
      b <- apply_scaler(XX[tr[iv], , drop = FALSE], sc)
      for (alpha in elastic_alpha_grid) {
        rows <- which(elastic_grid$alpha == alpha)
        fit <- glmnet::glmnet(a, y_model[tr[it]], alpha = alpha,
                              lambda = elastic_lambda_grid,
                              standardize = FALSE)
        pp <- as.matrix(predict(fit, newx = b,
                                s = elastic_grid$lambda[rows]))
        inner_error[rows, k] <- apply(pp, 2, function(v)
          rmse_vec(y_model[tr[iv]], v))
      }
    }
    scores <- rowMeans(inner_error)
    best <- which.min(scores)
    tuning_log[[paste0("ElasticNet_", f)]] <- data.frame(
      Outer_fold = f, elastic_grid, Inner_RMSE = scores,
      Selected = seq_len(nrow(elastic_grid)) == best)
    pred_elastic[va] <- fit_elastic_net(
      XX[tr, , drop = FALSE], y_model[tr], XX[va, , drop = FALSE],
      elastic_grid$alpha[best], elastic_grid$lambda[best]) / Y_SCALE
  }
  model_oof[["Elastic Net"]] <- pred_elastic
  
  # -----------------------------
  # 4J. OOF QC and tuning export
  # -----------------------------
  for (m in names(model_oof)) {
    if (length(model_oof[[m]]) != nrow(dat) ||
        any(!is.finite(model_oof[[m]]))) {
      stop("Invalid OOF predictions detected for model: ", m)
    }
  }
  
  oof_all <- data.frame(
    SampleID = dat$SampleID,
    River = dat$River,
    Observed_LCBD = y_all,
    Outer_fold = outer_fold_id,
    check.names = FALSE
  )
  
  for (m in names(model_oof)) {
    oof_all[[m]] <- model_oof[[m]]
  }
  
  write.csv(
    oof_all,
    file.path(OUT_DIR, "02A_all_models_OOF_predictions.csv"),
    row.names = FALSE
  )
  
  if (length(tuning_log) > 0) {
    tuning_export <- bind_rows(
      lapply(names(tuning_log), function(nm) {
        d <- tuning_log[[nm]]
        d$Tuning_block <- nm
        d
      })
    )
    
    write.csv(
      tuning_export,
      file.path(OUT_DIR, "02B_nested_CV_tuning_log.csv"),
      row.names = FALSE
    )
  }
  
  # -----------------------------
  # 4K. Auditable outer-fold model R2 and RFE feature frequency
  # -----------------------------
  # Fold R2 uses that fold's own observed mean. Pooled OOF R2 below is
  # calculated once over all samples and need not equal the mean of fold R2.
  fold_metrics <- do.call(rbind, lapply(names(model_oof), function(model) {
    do.call(rbind, lapply(seq_along(outer_folds), function(f) {
      va <- outer_folds[[f]]
      obs <- y_all[va]
      pred <- model_oof[[model]][va]
      data.frame(Model = model, Outer_fold = f, N = length(va),
                 R2 = calc_r2(obs, pred),
                 RMSE = sqrt(mean((obs - pred)^2)),
                 MAE = mean(abs(obs - pred)))
    }))
  }))
  write.csv(fold_metrics, file.path(OUT_DIR, "02C_model_per_outer_fold_metrics.csv"),
            row.names = FALSE)
  
  feature_frequency <- do.call(rbind, lapply(ml_models, function(model) {
    do.call(rbind, lapply(c(protected_core, env_candidates), function(variable) {
      present <- vapply(rfe_by_model[[model]],
                        function(z) variable %in% z$vars, logical(1))
      data.frame(Model = model, Predictor = variable,
                 Type = if (variable %in% protected_core) "Protected core" else "Environment",
                 Selected_folds = sum(present), Total_folds = length(present),
                 Selection_frequency = mean(present),
                 Selected_fold_IDs = paste(which(present), collapse = ";"))
    }))
  }))
  write.csv(feature_frequency,
            file.path(OUT_DIR, "01_model_specific_RFE_feature_frequency.csv"),
            row.names = FALSE)
  
  # Joint stability combines unsupervised fold-wise environmental screening and
  # model-specific RFE. A value of 1 means the optional variable survived both
  # stages in every outer fold for that model.
  joint_selection_stability <- feature_frequency %>%
    dplyr::filter(Type == "Environment") %>%
    dplyr::left_join(screening_frequency, by = "Predictor") %>%
    dplyr::mutate(
      Joint_selection_frequency = Selection_frequency,
      Stability_class = dplyr::case_when(
        Joint_selection_frequency >= 0.80 ~ "Highly stable",
        Joint_selection_frequency >= 0.60 ~ "Moderately stable",
        TRUE ~ "Unstable"
      )
    )
  write.csv(joint_selection_stability,
            file.path(OUT_DIR, "01_environment_joint_screen_RFE_stability.csv"),
            row.names = FALSE)
  
  # -----------------------------
  # 5. Model performance
  # -----------------------------
  performance <- bind_rows(
    lapply(names(model_oof), function(m) {
      cbind(
        Model = m,
        calc_metrics(y_all, model_oof[[m]])
      )
    })
  ) %>%
    arrange(desc(R2))
  
  write.csv(
    performance,
    file.path(OUT_DIR, "02_model_performance_9models.csv"),
    row.names = FALSE
  )
  
  best_model_name <- performance$Model[1]
  
  # Re-run redundancy filtering on all samples only for the final refit.
  # The winner's RFE on all samples uses the fixed outer folds as its
  # internal validation. These predictions are not reported as OOF performance.
  final_screen <- screen_predictors(dat, "full_data")
  full_rfe <- model_rfe(best_model_name, seq_len(nrow(dat)),
                        setdiff(final_screen$vars, protected_core), "full_data_final_refit", outer_fold_id)
  final_predictors <- full_rfe$vars
  selected_env <- setdiff(final_predictors, protected_core)
  write.csv(full_rfe$trace,
            file.path(OUT_DIR, "01_winner_full_data_RFE_trajectory.csv"), row.names = FALSE)
  write.csv(data.frame(Predictor = final_predictors,
                       Type = ifelse(final_predictors %in% protected_core, "Protected core", "Environment")),
            file.path(OUT_DIR, "01_final_refit_predictors.csv"), row.names = FALSE)
  write.csv(do.call(rbind, unlist(c(lapply(screen_by_fold, `[[`, "log"),
                                    list(final_screen$log)), recursive = FALSE)),
            file.path(OUT_DIR, "01_screening_decisions_and_core_exceptions.csv"), row.names = FALSE)
  cat("\nWinning model: ", best_model_name,
      "\nFinal refit predictors: ", paste(final_predictors, collapse = ", "),
      "\nAdditional environment: ",
      if (length(selected_env)) paste(selected_env, collapse = ", ") else "none",
      "\n", sep = "")
  X_all <- as.matrix(dat[, final_predictors, drop = FALSE])
  
  
  cat("\nBest ", HABITAT, " model by pooled OOF R2:", sep = "")
  cat(
    best_model_name, "\n")
  print(performance)
  
  # -----------------------------
  
  # Export full-data predictor correlation diagnostics.
  # These results are descriptive QC only and do not remove predictors.
  correlation_qc_full <- export_correlation_qc(
    dat = dat,
    predictors = c(protected_core, env_candidates),
    prefix = paste0("16S_", HABITAT),
    out_dir = OUT_DIR
  )
  
  # 6. Pentagonal radar plot
  # -----------------------------
  radar_metrics <- c(
    "R2",
    "1-nRMSE",
    "1-nMAE",
    "Pearson_r",
    "Explained_variance"
  )
  
  radar_labels <- c(
    "R²",
    "1−nRMSE",
    "1−nMAE",
    "Pearson r",
    "Expl. Var."
  )
  
  angles <- pi / 2 - 2 * pi * (0:4) / 5
  ring_levels <- seq(0.2, 1.0, by = 0.2)
  
  ring_df <- bind_rows(
    lapply(ring_levels, function(rr) {
      d <- data.frame(
        Ring = rr,
        Order = 1:5,
        x = rr * cos(angles),
        y = rr * sin(angles)
      )
      bind_rows(d, d[1, ])
    })
  )
  
  spoke_df <- data.frame(
    x = 0,
    y = 0,
    xend = cos(angles),
    yend = sin(angles)
  )
  
  radar_model <- performance %>%
    select(Model, all_of(radar_metrics)) %>%
    pivot_longer(
      -Model,
      names_to = "Metric",
      values_to = "Value"
    ) %>%
    mutate(
      Order = match(Metric, radar_metrics),
      Value = pmax(0, pmin(1, Value)),
      x = Value * cos(angles[Order]),
      y = Value * sin(angles[Order])
    ) %>%
    group_by(Model) %>%
    arrange(Order) %>%
    group_modify(~ bind_rows(.x, .x[1, ])) %>%
    ungroup()
  
  label_df <- data.frame(
    Label = radar_labels,
    x = 1.15 * cos(angles),
    y = 1.15 * sin(angles)
  )
  
  model_colors <- MODEL_COLORS_BASE
  
  line_types <- setNames(
    rep("solid", length(model_colors)),
    names(model_colors)
  )
  
  line_widths <- setNames(
    ifelse(
      names(model_colors) == best_model_name,
      1.45,
      0.70
    ),
    names(model_colors)
  )
  
  p_radar <- ggplot() +
    geom_path(
      data = ring_df,
      aes(x, y, group = Ring),
      color = "grey80",
      linetype = "dashed",
      linewidth = 0.45
    ) +
    geom_segment(
      data = spoke_df,
      aes(
        x = x,
        y = y,
        xend = xend,
        yend = yend
      ),
      color = "grey85",
      linewidth = 0.4
    ) +
    geom_path(
      data = radar_model,
      aes(
        x,
        y,
        group = Model,
        color = Model,
        linetype = Model,
        linewidth = Model
      )
    ) +
    geom_point(
      data = radar_model %>%
        group_by(Model) %>%
        slice_head(n = 5) %>%
        ungroup(),
      aes(x, y, color = Model),
      size = 2.1
    ) +
    geom_text(
      data = label_df,
      aes(x, y, label = Label),
      fontface = "bold",
      size = 4.1
    ) +
    scale_color_manual(values = model_colors) +
    scale_linetype_manual(values = line_types) +
    scale_linewidth_manual(values = line_widths) +
    coord_equal(
      xlim = c(-1.3, 1.3),
      ylim = c(-1.25, 1.28),
      clip = "off"
    ) +
    theme_void(base_size = 12) +
    theme(
      legend.position = "bottom",
      legend.title = element_text(face = "bold")
    ) +
    guides(
      linewidth = "none",
      linetype = "none",
      color = guide_legend(
        title = "Model types",
        nrow = 2,
        byrow = TRUE,
        override.aes = list(linewidth = 1.0, linetype = "solid")
      )
    )
  p_radar
  
  
  # -----------------------------
  # 7. Best-model OOF observed vs predicted
  # -----------------------------
  best_oof <- model_oof[[best_model_name]]
  
  best_metrics <- performance %>%
    filter(Model == best_model_name)
  
  best_prediction_table <- data.frame(
    SampleID = dat$SampleID,
    River = dat$River,
    Observed_LCBD = y_all,
    Predicted_LCBD_OOF = best_oof,
    Residual = y_all - best_oof,
    Outer_fold = outer_fold_id,
    stringsAsFactors = FALSE
  )
  
  write.csv(
    best_prediction_table,
    file.path(
      OUT_DIR,
      "03_best_model_OOF_observed_predicted.csv"
    ),
    row.names = FALSE
  )
  
  annotation_text <- sprintf(
    paste0(
      "Model: %s\n",
      "R² (OOF) = %.3f\n",
      "RMSE = %.6f\n",
      "MAE = %.6f\n",
      "Pearson r = %.3f"
    ),
    best_model_name,
    best_metrics$R2,
    best_metrics$RMSE,
    best_metrics$MAE,
    best_metrics$Pearson_r
  )
  
  common_range <- range(
    c(
      best_prediction_table$Observed_LCBD,
      best_prediction_table$Predicted_LCBD_OOF
    ),
    na.rm = TRUE
  )
  
  range_width <- diff(common_range)
  if (!is.finite(range_width) || range_width <= 0) range_width <- 1
  
  common_limits <- c(
    common_range[1] - 0.03 * range_width,
    common_range[2] + 0.03 * range_width
  )
  
  common_breaks <- pretty(common_limits, n = 5)
  common_breaks <- common_breaks[
    common_breaks >= common_limits[1] &
      common_breaks <= common_limits[2]
  ]
  
  annotation_x <- common_limits[1] +
    0.01 * diff(common_limits)
  
  annotation_y <- common_limits[2] -
    0.01 * diff(common_limits)
  
  p_obs <- ggplot(
    best_prediction_table,
    aes(
      x = Observed_LCBD,
      y = Predicted_LCBD_OOF
    )
  ) +
    geom_abline(
      slope = 1,
      intercept = 0,
      color = ONE_TO_ONE_COLOR,
      linetype = "dashed",
      linewidth = 0.85
    ) +
    geom_point(
      color = SCATTER_COLOR,
      size = 2.5,
      alpha = 0.88
    ) +
    annotate(
      "text",
      x = annotation_x,
      y = annotation_y,
      label = annotation_text,
      hjust = 0,
      vjust = 1,
      size = 3.6
    ) +
    scale_x_continuous(
      limits = common_limits,
      breaks = common_breaks,
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      limits = common_limits,
      breaks = common_breaks,
      expand = c(0, 0)
    ) +
    labs(
      x = "Observed overall LCBD",
      y = "Predicted overall LCBD"
    ) +
    coord_equal() +
    theme_classic(base_size = 12) +
    theme(
      axis.title = element_text(
        face = "bold",
        size = 13
      ),
      axis.text = element_text(
        color = "black"
      ),
      panel.border = element_rect(
        color = "black",
        fill = NA,
        linewidth = 0.8
      ),
      plot.margin = margin(6, 8, 6, 6)
    )
  p_obs
  
  # -----------------------------
  # 7B. DR auxiliary variables vs SAME best-model OOF prediction
  # -----------------------------
  dr_oof_table <- data.frame(
    SampleID = dat$SampleID,
    River = dat$River,
    Observed_Overall_LCBD = y_all,
    Predicted_Overall_LCBD_OOF = best_oof,
    DR_process = dat$DR_process,
    DR_LCBD = dat$DR_LCBD,
    DL_prediction_model = best_model_name,
    Outer_fold = outer_fold_id,
    stringsAsFactors = FALSE
  )
  
  write.csv(
    dr_oof_table,
    file.path(
      OUT_DIR,
      "03B_DR_auxiliary_with_best_model_OOF.csv"
    ),
    row.names = FALSE
  )
  
  format_p <- function(p) {
    if (!is.finite(p)) return("NA")
    if (p < 0.001) return("< 0.001")
    sprintf("= %.3f", p)
  }
  
  fit_dr_relation <- function(xvar, xlab) {
    d <- dr_oof_table[
      ,
      c(xvar, "Predicted_Overall_LCBD_OOF"),
      drop = FALSE
    ]
    
    names(d) <- c("x", "y")
    
    d <- d[
      is.finite(d$x) &
        is.finite(d$y),
      ,
      drop = FALSE
    ]
    
    if (nrow(d) < 3 ||
        length(unique(d$x)) < 2) {
      stop(
        "Insufficient finite/variable data for linear fit: ",
        xvar
      )
    }
    
    fit <- lm(y ~ x, data = d)
    sm <- summary(fit)
    ci <- confint(fit, level = 0.95)
    cf <- coef(sm)
    
    r <- cor(
      d$x,
      d$y,
      method = "pearson"
    )
    
    stat <- data.frame(
      Predictor = xvar,
      N = nrow(d),
      Intercept = unname(coef(fit)[1]),
      Slope = unname(coef(fit)[2]),
      Slope_SE = unname(cf["x", "Std. Error"]),
      Slope_CI95_low = unname(ci["x", 1]),
      Slope_CI95_high = unname(ci["x", 2]),
      Pearson_r = unname(r),
      R2 = sm$r.squared,
      Adjusted_R2 = sm$adj.r.squared,
      P_value = unname(cf["x", "Pr(>|t|)"]),
      stringsAsFactors = FALSE
    )
    
    x_range <- range(d$x, na.rm = TRUE)
    y_range <- range(d$y, na.rm = TRUE)
    
    x_width <- diff(x_range)
    y_width <- diff(y_range)
    
    if (!is.finite(x_width) || x_width <= 0) x_width <- 1
    if (!is.finite(y_width) || y_width <= 0) y_width <- 1
    
    ann_x <- x_range[1] + 0.04 * x_width
    ann_y <- y_range[2] - 0.04 * y_width
    
    ann <- sprintf(
      paste0(
        "Slope = %.4f\n",
        "R² = %.3f\n",
        "Pearson r = %.3f\n",
        "P %s\n",
        "N = %d"
      ),
      stat$Slope,
      stat$R2,
      stat$Pearson_r,
      format_p(stat$P_value),
      stat$N
    )
    
    p <- ggplot(
      d,
      aes(x = x, y = y)
    ) +
      geom_point(
        color = SCATTER_COLOR,
        size = 2.5,
        alpha = 0.88
      ) +
      geom_smooth(
        method = "lm",
        formula = y ~ x,
        se = TRUE,
        color = FIT_LINE_COLOR,
        fill = "grey80",
        linewidth = 0.9,
        alpha = 0.35
      ) +
      annotate(
        "text",
        x = ann_x,
        y = ann_y,
        label = ann,
        hjust = 0,
        vjust = 1,
        size = 3.6
      ) +
      labs(
        x = xlab,
        y = "OOF-predicted overall LCBD"
      ) +
      theme_classic(base_size = 12) +
      theme(
        axis.title = element_text(
          face = "bold",
          size = 13
        ),
        axis.text = element_text(
          color = "black"
        ),
        panel.border = element_rect(
          color = "black",
          fill = NA,
          linewidth = 0.8
        ),
        plot.margin = margin(6, 8, 6, 6)
      )
    
    list(
      stat = stat,
      plot = p
    )
  }
  
  dr_process_fit <- fit_dr_relation(
    "DR_process",
    "DR process importance"
  )
  
  dr_lcbd_fit <- fit_dr_relation(
    "DR_LCBD",
    "DR-ASV LCBD"
  )
  
  dr_fit_summary <- bind_rows(
    dr_process_fit$stat,
    dr_lcbd_fit$stat
  )
  
  write.csv(
    dr_fit_summary,
    file.path(
      OUT_DIR,
      "03C_DR_vs_best_model_OOF_linear_fit_summary.csv"
    ),
    row.names = FALSE
  )
  dr_process_fit$plot
  
  
  # -----------------------------
  # 8. Refit the winning model on all samples of the current habitat
  # -----------------------------
  # Full-data tuning uses the already-defined outer folds.
  full_cv_score <- function(predict_one_fold) {
    sapply(seq_along(outer_folds), function(f) {
      va <- outer_folds[[f]]
      tr <- setdiff(seq_len(nrow(dat)), va)
      pr <- predict_one_fold(tr, va, f)
      rmse_vec(y_model[va], pr)
    }) %>% mean()
  }
  
  final_model <- NULL
  final_model_type <- NULL
  final_scaler <- NULL
  final_extra <- list()
  
  if (best_model_name == "Linear Reg") {
    final_scaler <- fit_scaler(X_all)
    A <- apply_scaler(X_all, final_scaler)
    fit <- lm.fit(cbind(Intercept = 1, A), y_model)
    beta <- fit$coefficients
    beta[!is.finite(beta)] <- 0
    
    final_model <- beta
    final_model_type <- "linear"
    
    predict_final <- function(newX) {
      B <- apply_scaler(
        as.matrix(newX),
        final_scaler
      )
      as.numeric(
        cbind(Intercept = 1, B) %*% final_model
      ) / Y_SCALE
    }
    
  } else if (best_model_name == "Ridge") {
    score <- sapply(ridge_grid, function(a) {
      full_cv_score(function(tr, va, f) {
        fit_ridge(
          X_all[tr, , drop = FALSE],
          y_model[tr],
          X_all[va, , drop = FALSE],
          a
        )
      })
    })
    
    best_alpha <- ridge_grid[which.min(score)]
    final_scaler <- fit_scaler(X_all)
    A <- apply_scaler(X_all, final_scaler)
    ybar <- mean(y_model)
    beta <- solve(
      crossprod(A) + best_alpha * diag(ncol(A)),
      crossprod(A, y_model - ybar)
    )
    
    final_model <- list(
      beta = beta,
      ybar = ybar,
      alpha = best_alpha
    )
    final_model_type <- "ridge"
    
    predict_final <- function(newX) {
      B <- apply_scaler(
        as.matrix(newX),
        final_scaler
      )
      as.numeric(
        final_model$ybar +
          B %*% final_model$beta
      ) / Y_SCALE
    }
    
  } else if (best_model_name == "KNN") {
    score <- sapply(knn_grid, function(k) {
      full_cv_score(function(tr, va, f) {
        sc <- fit_scaler(X_all[tr, , drop = FALSE])
        A <- apply_scaler(X_all[tr, , drop = FALSE], sc)
        B <- apply_scaler(X_all[va, , drop = FALSE], sc)
        
        FNN::knn.reg(
          train = A,
          test = B,
          y = y_model[tr],
          k = k
        )$pred
      })
    })
    
    best_k <- knn_grid[which.min(score)]
    final_scaler <- fit_scaler(X_all)
    
    final_model <- list(
      X = apply_scaler(X_all, final_scaler),
      y = y_model,
      k = best_k
    )
    final_model_type <- "knn"
    
    predict_final <- function(newX) {
      B <- apply_scaler(
        as.matrix(newX),
        final_scaler
      )
      
      FNN::knn.reg(
        train = final_model$X,
        test = B,
        y = final_model$y,
        k = final_model$k
      )$pred / Y_SCALE
    }
    
  } else if (best_model_name == "SVR") {
    score <- rep(NA_real_, nrow(svr_grid))
    
    for (g in seq_len(nrow(svr_grid))) {
      score[g] <- full_cv_score(function(tr, va, f) {
        sc <- fit_scaler(X_all[tr, , drop = FALSE])
        A <- apply_scaler(X_all[tr, , drop = FALSE], sc)
        B <- apply_scaler(X_all[va, , drop = FALSE], sc)
        
        gam <- if (svr_grid$gamma_code[g] == "scale") {
          1 / ncol(A)
        } else {
          as.numeric(svr_grid$gamma_code[g])
        }
        
        fit <- e1071::svm(
          x = A,
          y = y_model[tr],
          type = "eps-regression",
          kernel = "radial",
          cost = svr_grid$C[g],
          gamma = gam,
          epsilon = svr_grid$epsilon[g],
          scale = FALSE
        )
        
        as.numeric(predict(fit, B))
      })
    }
    
    bp <- svr_grid[which.min(score), ]
    final_scaler <- fit_scaler(X_all)
    A <- apply_scaler(X_all, final_scaler)
    
    gam <- if (bp$gamma_code == "scale") {
      1 / ncol(A)
    } else {
      as.numeric(bp$gamma_code)
    }
    
    final_model <- e1071::svm(
      x = A,
      y = y_model,
      type = "eps-regression",
      kernel = "radial",
      cost = bp$C,
      gamma = gam,
      epsilon = bp$epsilon,
      scale = FALSE
    )
    
    final_model_type <- "svr"
    
    predict_final <- function(newX) {
      B <- apply_scaler(
        as.matrix(newX),
        final_scaler
      )
      
      as.numeric(
        predict(final_model, B)
      ) / Y_SCALE
    }
    
  } else if (best_model_name == "ANN") {
    score <- rep(NA_real_, nrow(ann_grid))
    
    for (g in seq_len(nrow(ann_grid))) {
      score[g] <- full_cv_score(function(tr, va, f) {
        fit_ann_safe(
          X_all[tr, , drop = FALSE],
          y_model[tr],
          X_all[va, , drop = FALSE],
          size = ann_grid$size[g],
          decay = ann_grid$decay[g],
          seed = 60000 + g * 100 + f
        )
      })
    }
    
    bp <- ann_grid[which.min(score), ]
    
    final_scaler <- fit_scaler(X_all)
    A <- apply_scaler(X_all, final_scaler)
    
    y_mu <- mean(y_model)
    y_sd <- sd(y_model)
    if (!is.finite(y_sd) || y_sd == 0) y_sd <- 1
    
    yz <- (y_model - y_mu) / y_sd
    
    set.seed(61000)
    
    final_model <- nnet::nnet(
      x = A,
      y = yz,
      size = bp$size,
      decay = bp$decay,
      linout = TRUE,
      trace = FALSE,
      maxit = 3000,
      MaxNWts = 10000
    )
    
    final_extra <- list(
      y_mu = y_mu,
      y_sd = y_sd
    )
    
    final_model_type <- "ann"
    
    predict_final <- function(newX) {
      B <- apply_scaler(
        as.matrix(newX),
        final_scaler
      )
      
      pz <- as.numeric(
        predict(final_model, B)
      )
      
      (
        final_extra$y_mu +
          final_extra$y_sd * pz
      ) / Y_SCALE
    }
    
  } else if (best_model_name == "RF") {
    score <- rep(NA_real_, nrow(rf_grid))
    
    for (g in seq_len(nrow(rf_grid))) {
      score[g] <- full_cv_score(function(tr, va, f) {
        fit <- ranger::ranger(
          x = as.data.frame(X_all[tr, , drop = FALSE]),
          y = y_model[tr],
          num.trees = 600,
          mtry = max(1L, min(ncol(X_all),
                             as.integer(round(ncol(X_all) * rf_grid$mtry_fraction[g])))),
          min.node.size = rf_grid$min.node.size[g],
          sample.fraction = rf_grid$sample.fraction[g],
          seed = 62000 + g * 100 + f,
          num.threads = 1
        )
        
        predict(
          fit,
          data = as.data.frame(
            X_all[va, , drop = FALSE]
          )
        )$predictions
      })
    }
    
    bp <- rf_grid[which.min(score), ]
    
    final_model <- ranger::ranger(
      x = as.data.frame(X_all),
      y = y_model,
      num.trees = 600,
      mtry = max(1L, min(ncol(X_all),
                         as.integer(round(ncol(X_all) * bp$mtry_fraction)))),
      min.node.size = bp$min.node.size,
      sample.fraction = bp$sample.fraction,
      seed = 63000,
      num.threads = 1
    )
    
    final_model_type <- "rf"
    
    predict_final <- function(newX) {
      predict(
        final_model,
        data = as.data.frame(newX)
      )$predictions / Y_SCALE
    }
    
  } else if (best_model_name == "XGBoost") {
    score <- rep(NA_real_, nrow(xgb_grid))
    
    for (g in seq_len(nrow(xgb_grid))) {
      score[g] <- full_cv_score(function(tr, va, f) {
        params <- list(
          objective = "reg:squarederror",
          eval_metric = "rmse",
          max_depth = as.integer(xgb_grid$max_depth[g]),
          learning_rate = xgb_grid$eta[g],
          gamma = xgb_grid$gamma[g],
          min_child_weight = xgb_grid$min_child_weight[g],
          subsample = xgb_grid$subsample[g],
          colsample_bytree = xgb_grid$colsample_bytree[g],
          lambda = xgb_grid$lambda[g],
          alpha = xgb_grid$alpha[g],
          tree_method = "hist",
          nthread = 1L,
          seed = as.integer(64000 + g * 100 + f)
        )
        
        fit <- xgboost::xgb.train(
          params = params,
          data = xgboost::xgb.DMatrix(
            X_all[tr, , drop = FALSE],
            label = y_model[tr]
          ),
          nrounds = as.integer(
            xgb_grid$nrounds[g]
          ),
          verbose = 0
        )
        
        predict(
          fit,
          xgboost::xgb.DMatrix(
            X_all[va, , drop = FALSE]
          )
        )
      })
    }
    
    bp <- xgb_grid[which.min(score), ]
    
    final_params <- list(
      objective = "reg:squarederror",
      eval_metric = "rmse",
      max_depth = as.integer(bp$max_depth),
      learning_rate = bp$eta,
      gamma = bp$gamma,
      min_child_weight = bp$min_child_weight,
      subsample = bp$subsample,
      colsample_bytree = bp$colsample_bytree,
      lambda = bp$lambda,
      alpha = bp$alpha,
      tree_method = "hist",
      nthread = 1L,
      seed = 65000L
    )
    
    final_model <- xgboost::xgb.train(
      params = final_params,
      data = xgboost::xgb.DMatrix(
        X_all,
        label = y_model
      ),
      nrounds = as.integer(bp$nrounds),
      verbose = 0
    )
    
    final_model_type <- "xgboost"
    
    predict_final <- function(newX) {
      as.numeric(
        predict(
          final_model,
          xgboost::xgb.DMatrix(
            as.matrix(newX)
          )
        )
      ) / Y_SCALE
    }
    
  } else if (best_model_name == "LightGBM") {
    score <- rep(NA_real_, nrow(lgb_grid))
    
    for (g in seq_len(nrow(lgb_grid))) {
      score[g] <- full_cv_score(function(tr, va, f) {
        params <- list(
          objective = "regression",
          metric = "rmse",
          num_leaves = as.integer(
            lgb_grid$num_leaves[g]
          ),
          max_depth = as.integer(
            lgb_grid$max_depth[g]
          ),
          learning_rate = lgb_grid$learning_rate[g],
          min_data_in_leaf = as.integer(
            lgb_grid$min_data_in_leaf[g]
          ),
          feature_fraction = lgb_grid$feature_fraction[g],
          bagging_fraction = lgb_grid$bagging_fraction[g],
          bagging_freq = ifelse(lgb_grid$bagging_fraction[g] < 1, 1L, 0L),
          lambda_l1 = lgb_grid$lambda_l1[g],
          lambda_l2 = lgb_grid$lambda_l2[g],
          seed = as.integer(
            66000 + g * 100 + f
          ),
          deterministic = TRUE,
          force_col_wise = TRUE,
          verbosity = -1,
          num_threads = 1L
        )
        
        fit <- lgb.train(
          params = params,
          data = lgb.Dataset(
            X_all[tr, , drop = FALSE],
            label = y_model[tr]
          ),
          nrounds = as.integer(lgb_grid$nrounds[g]),
          verbose = -1
        )
        
        predict(
          fit,
          X_all[va, , drop = FALSE]
        )
      })
    }
    
    bp <- lgb_grid[which.min(score), ]
    
    final_params <- list(
      objective = "regression",
      metric = "rmse",
      num_leaves = as.integer(bp$num_leaves),
      max_depth = as.integer(bp$max_depth),
      learning_rate = bp$learning_rate,
      min_data_in_leaf = as.integer(
        bp$min_data_in_leaf
      ),
      feature_fraction = bp$feature_fraction,
      bagging_fraction = bp$bagging_fraction,
      bagging_freq = ifelse(bp$bagging_fraction < 1, 1L, 0L),
      lambda_l1 = bp$lambda_l1,
      lambda_l2 = bp$lambda_l2,
      seed = 67000L,
      deterministic = TRUE,
      force_col_wise = TRUE,
      verbosity = -1,
      num_threads = 1L
    )
    
    final_model <- lgb.train(
      params = final_params,
      data = lgb.Dataset(
        X_all,
        label = y_model
      ),
      nrounds = as.integer(bp$nrounds),
      verbose = -1
    )
    
    final_model_type <- "lightgbm"
    
    predict_final <- function(newX) {
      as.numeric(
        predict(
          final_model,
          as.matrix(newX)
        )
      ) / Y_SCALE
    }
    
  } else if (best_model_name == "Elastic Net") {
    score <- vapply(seq_len(nrow(elastic_grid)), function(g) {
      full_cv_score(function(tr, va, f) {
        fit_elastic_net(X_all[tr, , drop = FALSE], y_model[tr],
                        X_all[va, , drop = FALSE],
                        elastic_grid$alpha[g], elastic_grid$lambda[g])
      })
    }, numeric(1))
    best <- which.min(score)
    final_scaler <- fit_scaler(X_all)
    a <- apply_scaler(X_all, final_scaler)
    final_model <- glmnet::glmnet(a, y_model,
                                  alpha = elastic_grid$alpha[best], lambda = elastic_grid$lambda[best],
                                  standardize = FALSE)
    final_extra <- as.list(elastic_grid[best, , drop = FALSE])
    final_model_type <- "elastic_net"
    predict_final <- function(newX) {
      b <- apply_scaler(as.matrix(newX), final_scaler)
      as.numeric(predict(final_model, newx = b,
                         s = final_extra$lambda)) / Y_SCALE
    }
  } else {
    stop("Unsupported best model: ", best_model_name)
  }
  
  # Audit the scaler used by the final all-sample refit. Trees use raw
  # predictor units; the other six models apply the training-only Z-score.
  if (is.null(final_scaler)) {
    scaling_audit <- data.frame(Model = best_model_name,
                                Predictor = final_predictors, Scaled = FALSE,
                                Training_mean = NA_real_, Training_sd = NA_real_)
  } else {
    scaling_audit <- data.frame(Model = best_model_name,
                                Predictor = final_predictors, Scaled = TRUE,
                                Training_mean = unname(final_scaler$mean[final_predictors]),
                                Training_sd = unname(final_scaler$sd[final_predictors]))
  }
  write.csv(scaling_audit,
            file.path(OUT_DIR, "04_final_refit_Zscore_parameters.csv"), row.names = FALSE)
  
  full_fit_pred <- predict_final(
    dat[, final_predictors, drop = FALSE]
  )
  
  write.csv(
    data.frame(
      SampleID = dat$SampleID,
      Observed_LCBD = y_all,
      Full_fit_predicted_LCBD = full_fit_pred
    ),
    file.path(
      OUT_DIR,
      "03F_best_model_full_data_refit_predictions.csv"
    ),
    row.names = FALSE
  )
  
  # -----------------------------
  # 9. SHAP analysis
  # -----------------------------
  if (final_model_type == "xgboost") {
    shap_full <- predict(
      final_model,
      xgboost::xgb.DMatrix(X_all),
      predcontrib = TRUE
    )
    
    shap_mat <- as.matrix(
      shap_full[
        ,
        seq_len(ncol(X_all)),
        drop = FALSE
      ]
    ) / Y_SCALE
    
  } else if (final_model_type == "lightgbm") {
    shap_full <- predict(
      final_model,
      X_all,
      type = "contrib"
    )
    
    shap_mat <- as.matrix(
      shap_full[
        ,
        seq_len(ncol(X_all)),
        drop = FALSE
      ]
    ) / Y_SCALE
    
  } else {
    X_shap <- as.data.frame(X_all)
    names(X_shap) <- final_predictors
    
    pred_wrapper <- function(object, newdata) {
      predict_final(
        newdata[, final_predictors, drop = FALSE]
      )
    }
    
    set.seed(68000)
    
    shap_mat <- fastshap::explain(
      object = final_model,
      X = X_shap,
      pred_wrapper = pred_wrapper,
      nsim = 200,
      adjust = TRUE
    )
    
    shap_mat <- as.matrix(shap_mat)
  }
  
  colnames(shap_mat) <- final_predictors
  
  shap_df <- as.data.frame(shap_mat)
  shap_df$SampleID <- dat$SampleID
  
  write.csv(
    shap_df,
    file.path(
      OUT_DIR,
      "04_SHAP_values_best_model.csv"
    ),
    row.names = FALSE
  )
  
  shap_importance <- data.frame(
    Predictor = colnames(shap_mat),
    Mean_abs_SHAP = colMeans(
      abs(shap_mat),
      na.rm = TRUE
    ),
    stringsAsFactors = FALSE
  ) %>%
    arrange(desc(Mean_abs_SHAP))
  
  write.csv(
    shap_importance,
    file.path(
      OUT_DIR,
      "04_SHAP_importance_best_model.csv"
    ),
    row.names = FALSE
  )
  
  feature_order <- rev(
    shap_importance$Predictor
  )
  
  feature_long <- as.data.frame(X_all) %>%
    setNames(final_predictors) %>%
    mutate(RowID = seq_len(n())) %>%
    pivot_longer(
      -RowID,
      names_to = "Predictor",
      values_to = "Feature_value"
    )
  
  shap_long <- as.data.frame(shap_mat) %>%
    mutate(RowID = seq_len(n())) %>%
    pivot_longer(
      -RowID,
      names_to = "Predictor",
      values_to = "SHAP"
    ) %>%
    left_join(
      feature_long,
      by = c("RowID", "Predictor")
    ) %>%
    group_by(Predictor) %>%
    mutate(
      Feature_color = if (
        dplyr::n_distinct(Feature_value) <= 1
      ) {
        rep(0.5, dplyr::n())
      } else {
        rng <- range(
          Feature_value,
          na.rm = TRUE
        )
        
        (
          Feature_value - rng[1]
        ) / (
          rng[2] - rng[1]
        )
      }
    ) %>%
    ungroup() %>%
    mutate(
      Predictor = factor(
        Predictor,
        levels = feature_order
      )
    )
  
  write.csv(
    shap_long %>%
      mutate(
        Predictor = as.character(Predictor)
      ),
    file.path(
      OUT_DIR,
      "04_SHAP_plot_data.csv"
    ),
    row.names = FALSE
  )
  
  feature_ranges <- feature_long %>%
    group_by(Predictor) %>%
    summarise(
      Observed_min = min(
        Feature_value,
        na.rm = TRUE
      ),
      Observed_max = max(
        Feature_value,
        na.rm = TRUE
      ),
      .groups = "drop"
    )
  
  write.csv(
    feature_ranges,
    file.path(
      OUT_DIR,
      "04_SHAP_feature_value_ranges.csv"
    ),
    row.names = FALSE
  )
  
  set.seed(42)
  
  p_shap <- ggplot(
    shap_long,
    aes(
      x = SHAP,
      y = Predictor,
      color = Feature_color
    )
  ) +
    geom_hline(
      yintercept = seq_along(feature_order),
      color = "grey82",
      linetype = "dotted",
      linewidth = 0.35
    ) +
    geom_vline(
      xintercept = 0,
      color = "grey45",
      linewidth = 0.55
    ) +
    geom_point(
      position = position_jitter(
        height = 0.095,
        width = 0,
        seed = 42
      ),
      size = 1.65,
      alpha = 0.95,
      stroke = 0
    ) +
    scale_color_gradientn(
      colors = c(
        SHAP_LOW_COLOR,
        "#7B61B4",
        SHAP_HIGH_COLOR
      ),
      values = c(0, 0.5, 1),
      limits = c(0, 1),
      breaks = c(0, 1),
      labels = c("Low", "High"),
      name = "Feature value",
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        barheight = grid::unit(
          30,
          "mm"
        ),
        barwidth = grid::unit(
          4.5,
          "mm"
        ),
        ticks = FALSE,
        frame.colour = "grey35"
      )
    ) +
    labs(
      x = "SHAP value",
      y = NULL,
      title = paste0("Predictive features for overall LCBD (", HABITAT, ")")
    ) +
    theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(
        face = "bold",
        hjust = 0.5,
        size = 14
      ),
      axis.title.x = element_text(
        face = "bold",
        size = 13
      ),
      axis.text.x = element_text(
        color = "black",
        size = 10
      ),
      axis.text.y = element_text(
        color = "black",
        size = 11
      ),
      legend.position = "right",
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 9),
      plot.margin = margin(6, 12, 6, 6)
    )
  p_shap
  
  
  # ============================================================
  # 10. Virtual screening: DL_LCBD x DL_process under two MAP-tail definitions
  #     Primary extreme scenario: lowest/highest 10% MAP
  #     Sensitivity scenario: lowest/highest 20% MAP
  #
  # IMPORTANT:
  # - The fitted winning model is NOT retrained for these scenarios.
  # - Each scenario preserves the observed environmental background of
  #   the real samples belonging to that MAP tail.
  # - ONLY DL_LCBD and DL_process are overwritten on the common grid.
  # - Raw winner-model predictions are the quantitative results.
  # - BAM is used only to smooth the raw surface for visualization.
  # ============================================================
  
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    install.packages("mgcv")
  }
  library(mgcv)
  
  # -----------------------------
  # 10A. Define MAP-tail scenarios
  # -----------------------------
  map_q10 <- as.numeric(quantile(dat$MAP, probs = 0.10, na.rm = TRUE))
  map_q20 <- as.numeric(quantile(dat$MAP, probs = 0.20, na.rm = TRUE))
  map_q80 <- as.numeric(quantile(dat$MAP, probs = 0.80, na.rm = TRUE))
  map_q90 <- as.numeric(quantile(dat$MAP, probs = 0.90, na.rm = TRUE))
  
  if (
    any(!is.finite(c(map_q10, map_q20, map_q80, map_q90))) ||
    !(map_q10 < map_q20 && map_q20 < map_q80 && map_q80 < map_q90)
  ) {
    stop("Invalid MAP P10/P20/P80/P90 thresholds in habitat: ", HABITAT)
  }
  
  map_scenarios <- list(
    `10%` = list(
      `Lowest 10% MAP` = which(dat$MAP <= map_q10),
      `Highest 10% MAP` = which(dat$MAP >= map_q90)
    ),
    `20%` = list(
      `Lowest 20% MAP` = which(dat$MAP <= map_q20),
      `Highest 20% MAP` = which(dat$MAP >= map_q80)
    )
  )
  
  for (tail_name in names(map_scenarios)) {
    for (scenario_name in names(map_scenarios[[tail_name]])) {
      if (length(map_scenarios[[tail_name]][[scenario_name]]) < 3) {
        stop(
          "Too few samples in MAP scenario: ",
          HABITAT, " / ", scenario_name
        )
      }
    }
  }
  
  # -----------------------------
  # 10B. Build one common 2D screening grid
  # -----------------------------
  # The same DL grid is used for the 10% and 20% analyses within a habitat.
  # Central 96% of the observed DL distributions is used to limit extreme
  # extrapolation while retaining the main empirical range.
  
  dl_lcbd_range <- as.numeric(
    quantile(dat$DL_LCBD, probs = c(0.02, 0.98), na.rm = TRUE)
  )
  
  dl_process_range <- as.numeric(
    quantile(dat$DL_process, probs = c(0.02, 0.98), na.rm = TRUE)
  )
  
  if (
    any(!is.finite(c(dl_lcbd_range, dl_process_range))) ||
    diff(dl_lcbd_range) <= 0 ||
    diff(dl_process_range) <= 0
  ) {
    stop("Invalid DL virtual-screening ranges in habitat: ", HABITAT)
  }
  
  GRID_N <- 120L
  
  screening_grid <- expand.grid(
    DL_LCBD = seq(
      dl_lcbd_range[1],
      dl_lcbd_range[2],
      length.out = GRID_N
    ),
    DL_process = seq(
      dl_process_range[1],
      dl_process_range[2],
      length.out = GRID_N
    ),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  
  # -----------------------------
  # 10C. Export real scenario backgrounds
  # -----------------------------
  scenario_background_samples <- dplyr::bind_rows(
    lapply(names(map_scenarios), function(tail_name) {
      dplyr::bind_rows(
        lapply(names(map_scenarios[[tail_name]]), function(scenario_name) {
          idx <- map_scenarios[[tail_name]][[scenario_name]]
          out <- dat[
            idx,
            unique(c("SampleID", "River", final_predictors)),
            drop = FALSE
          ]
          out$Tail_definition <- tail_name
          out$Scenario <- scenario_name
          out
        })
      )
    })
  )
  
  write.csv(
    scenario_background_samples,
    file.path(
      OUT_DIR,
      "05_virtual_screening_MAP_tail_background_samples.csv"
    ),
    row.names = FALSE
  )
  
  # -----------------------------
  # 10D. Baseline observed and model-predicted scenario contrasts
  # -----------------------------
  # No predictor is overwritten here. This QC distinguishes:
  # 1) the observed Low-vs-High MAP LCBD contrast,
  # 2) the fitted winner's baseline Low-vs-High contrast,
  # from the counterfactual virtual surfaces below.
  
  baseline_scenario_predictions <- dplyr::bind_rows(
    lapply(names(map_scenarios), function(tail_name) {
      dplyr::bind_rows(
        lapply(names(map_scenarios[[tail_name]]), function(scenario_name) {
          idx <- map_scenarios[[tail_name]][[scenario_name]]
          
          pred <- as.numeric(
            predict_final(
              dat[idx, final_predictors, drop = FALSE]
            )
          )
          
          data.frame(
            Tail_definition = tail_name,
            Scenario = scenario_name,
            SampleID = dat$SampleID[idx],
            Observed_LCBD = dat$Overall_LCBD[idx],
            Predicted_LCBD_baseline = pred,
            stringsAsFactors = FALSE
          )
        })
      )
    })
  )
  
  write.csv(
    baseline_scenario_predictions,
    file.path(
      OUT_DIR,
      "05_virtual_screening_BASELINE_MAP_tail_predictions.csv"
    ),
    row.names = FALSE
  )
  
  baseline_scenario_summary <- baseline_scenario_predictions %>%
    dplyr::group_by(Tail_definition, Scenario) %>%
    dplyr::summarise(
      N = dplyr::n(),
      Observed_LCBD_mean = mean(Observed_LCBD),
      Observed_LCBD_median = median(Observed_LCBD),
      Predicted_LCBD_baseline_mean = mean(Predicted_LCBD_baseline),
      Predicted_LCBD_baseline_median = median(Predicted_LCBD_baseline),
      .groups = "drop"
    )
  
  write.csv(
    baseline_scenario_summary,
    file.path(
      OUT_DIR,
      "05_virtual_screening_BASELINE_MAP_tail_summary.csv"
    ),
    row.names = FALSE
  )
  
  # -----------------------------
  # 10E. Batched scenario-specific virtual prediction
  # -----------------------------
  predict_virtual_scenario <- function(
    tail_name,
    scenario_name,
    scenario_idx,
    map_cutoff
  ) {
    scenario_background <- dat[
      scenario_idx,
      final_predictors,
      drop = FALSE
    ]
    
    n_background <- nrow(scenario_background)
    n_grid <- nrow(screening_grid)
    
    # For every grid point, repeat all real samples from the relevant
    # MAP-tail background. This performs the same g-computation average
    # as a sample-by-sample loop, but requires only one predict() call.
    grid_id <- rep(seq_len(n_grid), each = n_background)
    background_id <- rep(seq_len(n_background), times = n_grid)
    
    newdata <- scenario_background[
      background_id,
      ,
      drop = FALSE
    ]
    
    # Overwrite ONLY the two focal DL variables.
    newdata$DL_LCBD <- screening_grid$DL_LCBD[grid_id]
    newdata$DL_process <- screening_grid$DL_process[grid_id]
    
    # MAP, EC and every other retained predictor remain at the observed
    # value of the corresponding real scenario-background sample.
    pred_all <- as.numeric(
      predict_final(
        newdata[, final_predictors, drop = FALSE]
      )
    )
    
    if (
      length(pred_all) != nrow(newdata) ||
      any(!is.finite(pred_all))
    ) {
      stop(
        "Invalid virtual predictions: ",
        HABITAT, " / ", scenario_name
      )
    }
    
    pred_mean <- as.numeric(
      rowsum(pred_all, group = grid_id, reorder = FALSE)
    ) / n_background
    
    data.frame(
      Tail_definition = tail_name,
      Scenario = scenario_name,
      DL_LCBD = screening_grid$DL_LCBD,
      DL_process = screening_grid$DL_process,
      Predicted_LCBD_raw = pred_mean,
      MAP_cutoff = map_cutoff,
      MAP_n = n_background,
      Background_design =
        "scenario_specific_observed_environment_batched",
      stringsAsFactors = FALSE
    )
  }
  
  scenario_cutoffs <- list(
    `10%` = c(
      `Lowest 10% MAP` = map_q10,
      `Highest 10% MAP` = map_q90
    ),
    `20%` = c(
      `Lowest 20% MAP` = map_q20,
      `Highest 20% MAP` = map_q80
    )
  )
  
  vs_raw <- dplyr::bind_rows(
    lapply(names(map_scenarios), function(tail_name) {
      dplyr::bind_rows(
        lapply(names(map_scenarios[[tail_name]]), function(scenario_name) {
          predict_virtual_scenario(
            tail_name = tail_name,
            scenario_name = scenario_name,
            scenario_idx =
              map_scenarios[[tail_name]][[scenario_name]],
            map_cutoff =
              unname(scenario_cutoffs[[tail_name]][scenario_name])
          )
        })
      )
    })
  )
  
  write.csv(
    vs_raw,
    file.path(
      OUT_DIR,
      "05_virtual_screening_RAW_MAP_10pct_20pct.csv"
    ),
    row.names = FALSE
  )
  
  # -----------------------------
  # 10F. Quantitative Low-minus-High contrasts from RAW ML predictions
  # -----------------------------
  # These values, not the smoothed plotting surface, define the numerical
  # Low-vs-High MAP virtual-screening contrast.
  
  vs_contrast <- vs_raw %>%
    dplyr::group_by(Tail_definition, Scenario) %>%
    dplyr::summarise(
      Mean_predicted_LCBD = mean(Predicted_LCBD_raw),
      Median_predicted_LCBD = median(Predicted_LCBD_raw),
      Min_predicted_LCBD = min(Predicted_LCBD_raw),
      Max_predicted_LCBD = max(Predicted_LCBD_raw),
      MAP_cutoff = dplyr::first(MAP_cutoff),
      MAP_n = dplyr::first(MAP_n),
      .groups = "drop"
    )
  
  contrast_10 <- data.frame(
    Tail_definition = "10%",
    Low_mean = vs_contrast$Mean_predicted_LCBD[
      vs_contrast$Scenario == "Lowest 10% MAP"
    ],
    High_mean = vs_contrast$Mean_predicted_LCBD[
      vs_contrast$Scenario == "Highest 10% MAP"
    ]
  )
  contrast_10$Low_minus_High <- contrast_10$Low_mean - contrast_10$High_mean
  
  contrast_20 <- data.frame(
    Tail_definition = "20%",
    Low_mean = vs_contrast$Mean_predicted_LCBD[
      vs_contrast$Scenario == "Lowest 20% MAP"
    ],
    High_mean = vs_contrast$Mean_predicted_LCBD[
      vs_contrast$Scenario == "Highest 20% MAP"
    ]
  )
  contrast_20$Low_minus_High <- contrast_20$Low_mean - contrast_20$High_mean
  
  vs_low_high_contrast <- dplyr::bind_rows(
    contrast_10,
    contrast_20
  )
  
  write.csv(
    vs_contrast,
    file.path(
      OUT_DIR,
      "05_virtual_screening_RAW_scenario_summary.csv"
    ),
    row.names = FALSE
  )
  
  write.csv(
    vs_low_high_contrast,
    file.path(
      OUT_DIR,
      "05_virtual_screening_RAW_Low_minus_High_contrast.csv"
    ),
    row.names = FALSE
  )
  
  # -----------------------------
  # 10G. Direct winner-model virtual-screening surface
  # -----------------------------
  # No GAM/BAM/LOESS smoothing is applied.
  # The virtual-screening design already evaluates the winning model on a
  # regular 120 x 120 DL_LCBD x DL_process grid and averages predictions
  # equally across the observed MAP-tail background samples. Therefore, the
  # raw grid itself is the response surface used for visualization and all
  # quantitative summaries. This avoids introducing an additional smoother
  # that could alter the machine-learning response surface or fail numerically.
  vs_surface <- vs_raw
  vs_surface$Predicted_LCBD_surface <- vs_surface$Predicted_LCBD_raw
  vs_surface$Surface_method <- "raw_winner_model_grid"
  
  write.csv(
    vs_surface,
    file.path(
      OUT_DIR,
      "05_virtual_screening_SMOOTHED_MAP_10pct_20pct.csv"
    ),
    row.names = FALSE
  )
  
  # -----------------------------
  # 10H. Scenario metadata
  # -----------------------------
  scenario_metadata <- data.frame(
    Tail_definition = c("10%", "10%", "20%", "20%"),
    Scenario = c(
      "Lowest 10% MAP",
      "Highest 10% MAP",
      "Lowest 20% MAP",
      "Highest 20% MAP"
    ),
    MAP_cutoff = c(map_q10, map_q90, map_q20, map_q80),
    MAP_n = c(
      length(map_scenarios[["10%"]][["Lowest 10% MAP"]]),
      length(map_scenarios[["10%"]][["Highest 10% MAP"]]),
      length(map_scenarios[["20%"]][["Lowest 20% MAP"]]),
      length(map_scenarios[["20%"]][["Highest 20% MAP"]])
    ),
    DL_LCBD_min = dl_lcbd_range[1],
    DL_LCBD_max = dl_lcbd_range[2],
    DL_process_min = dl_process_range[1],
    DL_process_max = dl_process_range[2],
    Grid_n_per_axis = GRID_N,
    Grid_total_points = GRID_N^2,
    stringsAsFactors = FALSE
  )
  
  write.csv(
    scenario_metadata,
    file.path(
      OUT_DIR,
      "05_virtual_screening_MAP_10pct_20pct_metadata.csv"
    ),
    row.names = FALSE
  )
  
  # -----------------------------
  # 10I. Habitat-level 10% and 20% plots
  # -----------------------------
  # These are diagnostic habitat-level plots. The final cross-habitat
  # publication figures are generated after all three habitats finish.
  
  make_habitat_vs_plot <- function(tail_name) {
    d <- vs_surface[
      vs_surface$Tail_definition == tail_name,
      ,
      drop = FALSE
    ]
    
    scenario_order <- if (tail_name == "10%") {
      c("Lowest 10% MAP", "Highest 10% MAP")
    } else {
      c("Lowest 20% MAP", "Highest 20% MAP")
    }
    
    meta <- scenario_metadata[
      scenario_metadata$Tail_definition == tail_name,
      ,
      drop = FALSE
    ]
    
    labels <- setNames(
      paste0(
        meta$Scenario,
        "\n",
        ifelse(grepl("^Lowest", meta$Scenario), "MAP <= ", "MAP >= "),
        format(round(meta$MAP_cutoff, 1), nsmall = 1),
        " (n = ", meta$MAP_n, ")"
      ),
      meta$Scenario
    )
    
    d$Scenario <- factor(d$Scenario, levels = scenario_order)
    
    ggplot2::ggplot(
      d,
      ggplot2::aes(x = DL_LCBD, y = DL_process)
    ) +
      ggplot2::geom_raster(
        ggplot2::aes(fill = Predicted_LCBD_surface),
        interpolate = TRUE
      ) +
      ggplot2::geom_contour(
        ggplot2::aes(z = Predicted_LCBD_surface),
        color = "grey45",
        linewidth = 0.25,
        alpha = 0.70
      ) +
      ggplot2::facet_wrap(
        ~ Scenario,
        nrow = 1,
        labeller = ggplot2::as_labeller(labels)
      ) +
      ggplot2::scale_fill_gradientn(
        colours = c(
          "#2166AC", "#67A9CF", "#D1E5F0",
          "#F7F7F7",
          "#FDDBC7", "#EF8A62", "#B2182B"
        ),
        name = "Predicted overall LCBD"
      ) +
      ggplot2::scale_x_continuous(expand = c(0, 0)) +
      ggplot2::scale_y_continuous(expand = c(0, 0)) +
      ggplot2::labs(
        title = paste0(HABITAT, " | MAP tails: ", tail_name),
        x = "DL-LCBD",
        y = "DL process"
      ) +
      ggplot2::theme_classic(base_size = 12) +
      ggplot2::theme(
        plot.title = ggplot2::element_text(
          face = "bold", hjust = 0.5
        ),
        strip.background = ggplot2::element_blank(),
        strip.text = ggplot2::element_text(face = "bold"),
        panel.border = ggplot2::element_rect(
          color = "black", fill = NA, linewidth = 0.55
        )
      )
  }
  
  p_vs_10_habitat <- make_habitat_vs_plot("10%")
  p_vs_20_habitat <- make_habitat_vs_plot("20%")
  
  ggsave(
    file.path(
      OUT_DIR,
      "05_virtual_screening_MAP_10pct.pdf"
    ),
    p_vs_10_habitat,
    width = 10.2,
    height = 5.2,
    bg = "white"
  )
  
  ggsave(
    file.path(
      OUT_DIR,
      "05_virtual_screening_MAP_20pct.pdf"
    ),
    p_vs_20_habitat,
    width = 10.2,
    height = 5.2,
    bg = "white"
  )
  
  
  summary_lines <- c(
    paste0(
      paste0("Habitat: ", HABITAT)
    ),
    paste0(
      "N samples: ",
      nrow(dat)
    ),
    paste0(
      "Input file: ",
      INPUT_FILE
    ),
    paste0(
      "Protected core: ",
      paste(
        protected_core,
        collapse = ", "
      )
    ),
    paste0(
      "Auxiliary variables excluded from ML: ",
      paste(
        auxiliary_vars,
        collapse = ", "
      )
    ),
    paste0(
      "Selected environmental predictors: ",
      ifelse(
        length(selected_env) == 0,
        "None",
        paste(
          selected_env,
          collapse = ", "
        )
      )
    ),
    paste0(
      "Final predictors: ",
      paste(
        final_predictors,
        collapse = ", "
      )
    ),
    paste0(
      "Best model: ",
      best_model_name
    ),
    paste0(
      "Best OOF R2: ",
      round(
        best_metrics$R2,
        4
      )
    ),
    paste0(
      "Best OOF RMSE: ",
      signif(
        best_metrics$RMSE,
        6
      )
    ),
    paste0(
      "Best OOF MAE: ",
      signif(
        best_metrics$MAE,
        6
      )
    ),
    paste0(
      "Best OOF Pearson r: ",
      round(
        best_metrics$Pearson_r,
        4
      )
    ),
    paste0(
      "Low MAP group threshold P10: ",
      round(
        map_q10,
        3
      )
    ),
    paste0(
      "High MAP group threshold P90: ",
      round(
        map_q90,
        3
      )
    ),
    paste0(
      "Low MAP sensitivity threshold P20: ",
      round(
        map_q20,
        3
      )
    ),
    paste0(
      "High MAP sensitivity threshold P80: ",
      round(
        map_q80,
        3
      )
    ),
    paste0(
      "Scatter color: ",
      SCATTER_COLOR
    ),
    "DR_process and DR_LCBD were excluded from spatialRF screening, model-specific RFE, all nine ML models, SHAP and virtual screening.",
    "DR linear fits use the exact best-model OOF predictions generated in this same run.",
    "Observed-vs-predicted plot uses identical x/y limits and tick positions.",
    "Model performance is based only on pooled outer-fold OOF predictions.",
    "SVR, ANN, RF, XGBoost and LightGBM use expanded inner-CV tuning spaces; outer test folds are never used for hyperparameter selection.",
    "Full-data refit is used only for SHAP and virtual screening.",
    "Virtual screening uses scenario-specific observed environmental backgrounds: only DL_LCBD and DL_process are overwritten; MAP and all other retained predictors remain at each scenario sample\'s observed values.",
    "Predictions are averaged equally across the real samples within each MAP-decile scenario.",
    "No secondary GAM/BAM smoothing is applied; virtual-screening figures and scenario contrasts use the raw winner-model prediction grid directly."
  )
  
  writeLines(
    summary_lines,
    file.path(
      OUT_DIR,
      "00_analysis_summary.txt"
    )
  )
  
  cat(
    paste(
      summary_lines,
      collapse = "\n"
    ),
    "\n"
  )
  
  
  # Return objects needed for cross-habitat combined figures.
  list(
    Habitat = HABITAT,
    performance = performance,
    best_model_name = best_model_name,
    best_metrics = best_metrics,
    best_prediction_table = best_prediction_table,
    p_radar = p_radar,
    p_obs = p_obs,
    p_shap = p_shap,
    p_vs = p_vs_continuous,
    shap_long = shap_long,
    vs_surface = vs_surface,
    surface_limits = surface_limits,
    scenario_labels = scenario_labels,
    scenario_metadata = scenario_metadata,
    vs_low_high_contrast = vs_low_high_contrast,
    baseline_scenario_summary = baseline_scenario_summary,
    dr_oof_table = dr_oof_table,
    dr_fit_summary = dr_fit_summary,
    final_predictors = final_predictors
  )
  
}


# ============================================================
# Configuration audit before model fitting
# ============================================================
configuration_audit <- data.frame(
  Habitat = names(INPUT_FILES),
  Input_file = unname(INPUT_FILES),
  Output_directory = file.path(MAIN_OUT_DIR, names(INPUT_FILES)),
  File_exists = file.exists(unname(INPUT_FILES)),
  stringsAsFactors = FALSE
)

write.csv(
  configuration_audit,
  file.path(MAIN_OUT_DIR, "00_COMBINED_input_output_configuration.csv"),
  row.names = FALSE
)

if (any(!configuration_audit$File_exists)) {
  stop(
    "One or more input files are missing: ",
    paste(
      configuration_audit$Input_file[!configuration_audit$File_exists],
      collapse = ", "
    )
  )
}

# ============================================================
# Run all three habitats independently
# ============================================================
results <- lapply(names(INPUT_FILES), function(h) {
  message("\n============================================================")
  message("Running 16S habitat: ", h)
  message("============================================================")
  run_habitat(
    HABITAT = h,
    INPUT_FILE = unname(INPUT_FILES[h]),
    MAIN_OUT_DIR = MAIN_OUT_DIR
  )
})
names(results) <- names(INPUT_FILES)

# -----------------------------
# Combined performance table
# -----------------------------
combined_performance <- dplyr::bind_rows(
  lapply(names(results), function(h) {
    transform(results[[h]]$performance, Habitat = h)
  })
) %>%
  dplyr::select(Habitat, dplyr::everything())

write.csv(
  combined_performance,
  file.path(MAIN_OUT_DIR, "COMBINED_16S_model_performance.csv"),
  row.names = FALSE
)

best_model_summary <- dplyr::bind_rows(
  lapply(names(results), function(h) {
    z <- results[[h]]$best_metrics
    data.frame(
      Habitat = h,
      Best_model = results[[h]]$best_model_name,
      R2 = z$R2,
      RMSE = z$RMSE,
      MAE = z$MAE,
      Pearson_r = z$Pearson_r,
      Explained_variance = z$Explained_variance,
      Final_predictors = paste(results[[h]]$final_predictors, collapse = ";"),
      stringsAsFactors = FALSE
    )
  })
)
write.csv(
  best_model_summary,
  file.path(MAIN_OUT_DIR, "COMBINED_16S_best_model_summary.csv"),
  row.names = FALSE
)

# ============================================================
# Combined Figure 1: radar plots
# One common model legend
# ============================================================
radar_panels <- lapply(names(results), function(h) {
  results[[h]]$p_radar +
    ggplot2::labs(
      title = h,
      subtitle = paste0("Best: ", results[[h]]$best_model_name)
    ) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(
        face = "bold", hjust = 0.5, size = 14
      ),
      plot.subtitle = ggplot2::element_text(
        hjust = 0.5, size = 10
      ),
      legend.position = "bottom"
    )
})

p_radar_combined <- patchwork::wrap_plots(
  radar_panels,
  nrow = 1,
  guides = "collect"
) &
  ggplot2::theme(legend.position = "bottom")

print(p_radar_combined)

ggsave(
  file.path(MAIN_OUT_DIR, "COMBINED_16S_radar_3habitats.pdf"),
  p_radar_combined,
  width = 14.5, height = 5.2, bg = "white"
)
ggsave(
  file.path(MAIN_OUT_DIR, "COMBINED_16S_radar_3habitats.png"),
  p_radar_combined,
  width = 14.5, height = 5.2, dpi = 600, bg = "white"
)

# ============================================================
# Combined Figure 2: best-model OOF observed vs predicted
# Same physical panel width/height; habitat-specific numeric ranges retained.
# ============================================================
fit_panels <- lapply(names(results), function(h) {
  results[[h]]$p_obs +
    ggplot2::labs(title = h) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(
        face = "bold", hjust = 0.5, size = 14
      )
    )
})

p_fit_combined <- patchwork::wrap_plots(fit_panels, nrow = 1)
print(p_fit_combined)

ggsave(
  file.path(MAIN_OUT_DIR, "COMBINED_16S_OOF_fit_3habitats.pdf"),
  p_fit_combined,
  width = 12.6, height = 4.3, bg = "white"
)
ggsave(
  file.path(MAIN_OUT_DIR, "COMBINED_16S_OOF_fit_3habitats.png"),
  p_fit_combined,
  width = 12.6, height = 4.3, dpi = 600, bg = "white"
)

# ============================================================
# Combined Figure 3: downstream DR relationships
# DR variables are NOT ML predictors.
# The response is the SAME outer-fold OOF prediction from the winning model.
# Rows: DR_process and DR_LCBD; columns: Water, Sediment and Soil.
# A common y-axis is used across all six panels for direct comparison.
# ============================================================

combined_dr_oof <- dplyr::bind_rows(
  lapply(names(results), function(h) {
    d <- results[[h]]$dr_oof_table
    d$Habitat <- h
    d
  })
)

combined_dr_stats <- dplyr::bind_rows(
  lapply(names(results), function(h) {
    d <- results[[h]]$dr_fit_summary
    d$Habitat <- h
    d
  })
) %>%
  dplyr::select(Habitat, dplyr::everything())

write.csv(
  combined_dr_oof,
  file.path(MAIN_OUT_DIR, "COMBINED_16S_DR_auxiliary_with_best_model_OOF.csv"),
  row.names = FALSE
)

write.csv(
  combined_dr_stats,
  file.path(MAIN_OUT_DIR, "COMBINED_16S_DR_vs_best_model_OOF_linear_fit_summary.csv"),
  row.names = FALSE
)

dr_y_limits <- range(
  combined_dr_oof$Predicted_Overall_LCBD_OOF,
  na.rm = TRUE
)
dr_y_pad <- diff(dr_y_limits) * 0.04
if (!is.finite(dr_y_pad) || dr_y_pad <= 0) dr_y_pad <- 0
dr_y_limits <- c(
  dr_y_limits[1] - dr_y_pad,
  dr_y_limits[2] + dr_y_pad
)

format_combined_p <- function(p) {
  if (!is.finite(p)) return("NA")
  if (p < 0.001) return("< 0.001")
  sprintf("= %.3f", p)
}

make_combined_dr_panel <- function(h, xvar, xlab) {
  d <- results[[h]]$dr_oof_table[
    ,
    c(xvar, "Predicted_Overall_LCBD_OOF"),
    drop = FALSE
  ]
  names(d) <- c("x", "y")
  d <- d[is.finite(d$x) & is.finite(d$y), , drop = FALSE]
  
  st <- results[[h]]$dr_fit_summary
  st <- st[st$Predictor == xvar, , drop = FALSE]
  
  xr <- range(d$x, na.rm = TRUE)
  xw <- diff(xr)
  if (!is.finite(xw) || xw <= 0) xw <- 1
  
  ann_x <- xr[1] + 0.04 * xw
  ann_y <- dr_y_limits[2] - 0.04 * diff(dr_y_limits)
  
  ann <- sprintf(
    "Slope = %.4f\nR² = %.3f\nP %s",
    st$Slope,
    st$R2,
    format_combined_p(st$P_value)
  )
  
  ggplot2::ggplot(d, ggplot2::aes(x = x, y = y)) +
    ggplot2::geom_point(
      color = unname(HABITAT_COLORS_GLOBAL[h]),
      size = 2.2,
      alpha = 0.86
    ) +
    ggplot2::geom_smooth(
      method = "lm",
      formula = y ~ x,
      se = TRUE,
      color = unname(FIT_LINE_COLORS_GLOBAL[h]),
      fill = "grey82",
      linewidth = 0.9,
      alpha = 0.35
    ) +
    ggplot2::annotate(
      "text",
      x = ann_x,
      y = ann_y,
      label = ann,
      hjust = 0,
      vjust = 1,
      size = 3.4
    ) +
    ggplot2::coord_cartesian(ylim = dr_y_limits) +
    ggplot2::labs(
      x = xlab,
      y = if (h == "Water") "OOF-predicted overall LCBD" else NULL,
      title = if (xvar == "DR_process") h else NULL
    ) +
    ggplot2::theme_classic(base_size = 11.5) +
    ggplot2::theme(
      axis.title = ggplot2::element_text(face = "bold", size = 12),
      axis.text = ggplot2::element_text(color = "black"),
      plot.title = ggplot2::element_text(
        face = "bold", hjust = 0.5, size = 13
      ),
      panel.border = ggplot2::element_rect(
        color = "black", fill = NA, linewidth = 0.8
      ),
      plot.margin = ggplot2::margin(6, 7, 6, 6)
    )
}

dr_process_panels <- lapply(
  names(results),
  function(h) make_combined_dr_panel(
    h,
    "DR_process",
    "DR process importance"
  )
)

dr_lcbd_panels <- lapply(
  names(results),
  function(h) make_combined_dr_panel(
    h,
    "DR_LCBD",
    "DR-ASV LCBD"
  )
)

p_dr_combined <- patchwork::wrap_plots(
  c(dr_process_panels, dr_lcbd_panels),
  ncol = 3,
  byrow = TRUE
) +
  patchwork::plot_annotation(
    tag_levels = "A"
  )

print(p_dr_combined)




# ============================================================
# Combined Figure 4: SHAP beeswarm
# Shared Feature-value legend.
# IMPORTANT: x-axis panel widths are kept identical, but x scales remain
# habitat-specific because SHAP magnitudes can legitimately differ.
# Panel heights are proportional to the number of selected predictors so
# Water/Sediment/Soil do not need identical vertical heights.
# ============================================================
shap_panels <- lapply(names(results), function(h) {
  results[[h]]$p_shap +
    ggplot2::labs(title = h) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(
        face = "bold", hjust = 0.5, size = 13
      ),
      legend.position = "right"
    )
})

# Extract one common SHAP legend because all three panels use the same
# normalized feature-value color scale (Low -> High).
shap_legend <- cowplot::get_legend(
  shap_panels[[1]] + ggplot2::theme(legend.position = "right")
)
shap_no_legend <- lapply(
  shap_panels,
  function(p) p + ggplot2::theme(legend.position = "none")
)

shap_n <- vapply(
  results,
  function(z) length(unique(as.character(z$shap_long$Predictor))),
  numeric(1)
)
shap_rel_heights <- pmax(shap_n, 4)

# Stack vertically: identical plot width / x-axis physical length;
# each habitat receives height according to its number of SHAP predictors.
shap_stack <- cowplot::plot_grid(
  plotlist = shap_no_legend,
  ncol = 1,
  align = "v",
  axis = "lr",
  rel_heights = shap_rel_heights
)

p_shap_combined <- cowplot::plot_grid(
  shap_stack,
  shap_legend,
  nrow = 1,
  rel_widths = c(1, 0.12)
)

print(p_shap_combined)

shap_total_height <- max(8.5, sum(shap_rel_heights) * 0.43)

# ============================================================
# Combined Figures 5-6: MAP-tail virtual screening
#
# Figure 5: Lowest/Highest 10% MAP, arranged as 2 rows x 3 habitats
# Figure 6: Lowest/Highest 20% MAP, arranged as 2 rows x 3 habitats
#
# IMPORTANT:
# Both figures use ONE global predicted-LCBD color scale calculated from
# all 12 surfaces (3 habitats x 2 tail definitions x 2 scenarios).
# Therefore color intensity is directly comparable between the 10% and
# 20% sensitivity analyses as well as among habitats.
# ============================================================

vs_all <- dplyr::bind_rows(
  lapply(names(results), function(h) {
    d <- results[[h]]$vs_surface
    d$Habitat <- h
    d
  })
)

vs_meta_all <- dplyr::bind_rows(
  lapply(names(results), function(h) {
    d <- results[[h]]$scenario_metadata
    d$Habitat <- h
    d
  })
)

vs_all$Habitat <- factor(
  vs_all$Habitat,
  levels = c("Water", "Sediment", "Soil")
)

global_vs_limits <- range(
  vs_all$Predicted_LCBD_surface,
  finite = TRUE
)

if (
  length(global_vs_limits) != 2 ||
  any(!is.finite(global_vs_limits)) ||
  diff(global_vs_limits) <= 0
) {
  stop("Invalid global virtual-screening color limits.")
}

global_vs_breaks <- pretty(global_vs_limits, n = 7)

make_combined_vs_panel <- function(h, tail_name, scenario_name) {
  d <- vs_all[
    as.character(vs_all$Habitat) == h &
      vs_all$Tail_definition == tail_name &
      as.character(vs_all$Scenario) == scenario_name,
    ,
    drop = FALSE
  ]
  
  meta <- vs_meta_all[
    vs_meta_all$Habitat == h &
      vs_meta_all$Tail_definition == tail_name &
      vs_meta_all$Scenario == scenario_name,
    ,
    drop = FALSE
  ]
  
  if (nrow(d) == 0 || nrow(meta) != 1) {
    stop(
      "Missing combined virtual-screening data: ",
      h, " / ", tail_name, " / ", scenario_name
    )
  }
  
  cutoff_text <- if (grepl("^Lowest", scenario_name)) {
    paste0("MAP <= ", format(round(meta$MAP_cutoff, 1), nsmall = 1))
  } else {
    paste0("MAP >= ", format(round(meta$MAP_cutoff, 1), nsmall = 1))
  }
  
  panel_subtitle <- paste0(
    cutoff_text,
    " (n = ", meta$MAP_n, ")"
  )
  
  ggplot2::ggplot(
    d,
    ggplot2::aes(x = DL_LCBD, y = DL_process)
  ) +
    ggplot2::geom_raster(
      ggplot2::aes(fill = Predicted_LCBD_surface),
      interpolate = TRUE
    ) +
    ggplot2::geom_contour(
      ggplot2::aes(z = Predicted_LCBD_surface),
      breaks = global_vs_breaks,
      color = "grey45",
      linewidth = 0.25,
      alpha = 0.70
    ) +
    ggplot2::scale_fill_gradientn(
      colours = c(
        "#2166AC", "#67A9CF", "#D1E5F0",
        "#F7F7F7",
        "#FDDBC7", "#EF8A62", "#B2182B"
      ),
      limits = global_vs_limits,
      breaks = global_vs_breaks,
      name = "Predicted overall LCBD"
    ) +
    ggplot2::scale_x_continuous(expand = c(0, 0)) +
    ggplot2::scale_y_continuous(expand = c(0, 0)) +
    ggplot2::labs(
      title = h,
      subtitle = panel_subtitle,
      x = "DL-LCBD",
      y = "DL process"
    ) +
    ggplot2::theme_classic(base_size = 11.5) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(
        face = "bold", hjust = 0.5, size = 12.5
      ),
      plot.subtitle = ggplot2::element_text(
        hjust = 0.5, size = 9.8
      ),
      axis.title = ggplot2::element_text(face = "bold"),
      panel.border = ggplot2::element_rect(
        color = "black", fill = NA, linewidth = 0.55
      ),
      legend.position = "right"
    )
}

make_combined_vs_figure <- function(tail_name) {
  low_name <- if (tail_name == "10%") {
    "Lowest 10% MAP"
  } else {
    "Lowest 20% MAP"
  }
  
  high_name <- if (tail_name == "10%") {
    "Highest 10% MAP"
  } else {
    "Highest 20% MAP"
  }
  
  habitats <- c("Water", "Sediment", "Soil")
  
  low_panels <- lapply(
    habitats,
    function(h) make_combined_vs_panel(
      h, tail_name, low_name
    )
  )
  
  high_panels <- lapply(
    habitats,
    function(h) make_combined_vs_panel(
      h, tail_name, high_name
    )
  )
  
  # 2 rows x 3 columns:
  # row 1 = Lowest MAP tail
  # row 2 = Highest MAP tail
  patchwork::wrap_plots(
    c(low_panels, high_panels),
    ncol = 3,
    byrow = TRUE,
    guides = "collect"
  ) +
    patchwork::plot_annotation(
      title = paste0(
        "16S virtual screening | MAP tails: ",
        tail_name
      ),
      subtitle = paste0(
        "Top row: lowest ", tail_name,
        " MAP; bottom row: highest ", tail_name, " MAP"
      ),
      tag_levels = "A"
    ) &
    ggplot2::theme(
      legend.position = "right",
      plot.title = ggplot2::element_text(
        face = "bold", hjust = 0.5
      ),
      plot.subtitle = ggplot2::element_text(
        hjust = 0.5
      )
    )
}

p_vs_combined_10 <- make_combined_vs_figure("10%")
p_vs_combined_20 <- make_combined_vs_figure("20%")

print(p_vs_combined_10)
print(p_vs_combined_20)





# -----------------------------
# Combined quantitative virtual-screening summary
# -----------------------------
combined_vs_contrast <- dplyr::bind_rows(
  lapply(names(results), function(h) {
    d <- results[[h]]$vs_low_high_contrast
    d$Habitat <- h
    d
  })
) %>%
  dplyr::select(
    Habitat,
    Tail_definition,
    Low_mean,
    High_mean,
    Low_minus_High
  )

combined_baseline_summary <- dplyr::bind_rows(
  lapply(names(results), function(h) {
    d <- results[[h]]$baseline_scenario_summary
    d$Habitat <- h
    d
  })
) %>%
  dplyr::select(Habitat, dplyr::everything())

write.csv(
  combined_vs_contrast,
  file.path(
    MAIN_OUT_DIR,
    "COMBINED_16S_virtual_screening_RAW_Low_minus_High_contrast.csv"
  ),
  row.names = FALSE
)

write.csv(
  combined_baseline_summary,
  file.path(
    MAIN_OUT_DIR,
    "COMBINED_16S_virtual_screening_BASELINE_summary.csv"
  ),
  row.names = FALSE
)

# -----------------------------
# Final combined-analysis note
# -----------------------------
writeLines(
  c(
    "16S Water/Sediment/Soil integrated workflow completed.",
    "The three habitats were modeled independently rather than pooled into one model.",
    "All habitats used the same protected-core, model-specific RFE, nested-CV, SHAP and scenario-specific virtual-screening logic.",
    "DR_process and DR_LCBD remain downstream-only variables and are related to the same winner-model outer-fold OOF predictions.",
    "Virtual screening uses the already fitted full-data winner; the ML model is not retrained for the 10% and 20% MAP-tail analyses.",
    "The primary extreme scenario compares MAP <= P10 with MAP >= P90.",
    "The sensitivity scenario compares MAP <= P20 with MAP >= P80.",
    "Within each scenario, only DL_LCBD and DL_process are overwritten; MAP, EC and every other retained predictor preserve each real scenario sample's observed value.",
    "Virtual predictions are averaged equally across the real samples in each MAP-tail background.",
    "A 120 x 120 DL grid and batched prediction are used for computational efficiency without changing the estimand.",
    "BAM smoothing is visualization-only; Low-minus-High contrasts are calculated from raw winner-model predictions.",
    "The 10% and 20% combined virtual-screening figures are both arranged as 2 rows x 3 habitats.",
    "Both virtual-screening figures share one global predicted-LCBD color scale across all 12 surfaces."
  ),
  file.path(
    MAIN_OUT_DIR,
    "COMBINED_16S_analysis_note.txt"
  )
)

message(
  "\nAll three 16S habitat analyses and all six combined figure classes are complete."
)
