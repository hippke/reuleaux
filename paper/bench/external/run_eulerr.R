# eulerr (Larsson & Gustafsson 2018), exact ellipse/circle region areas,
# internal routine intersect_ellipses(par = c(h,k,r per set), circle = TRUE).
# The 7th entry (sets A, B, C jointly, exclusive) is the triple overlap.
library(eulerr)
f <- eulerr:::intersect_ellipses
X <- as.matrix(read.csv("configs.csv", header = FALSE))
tri <- function(r) f(r, TRUE, FALSE)[7]
out <- apply(X, 1, function(r) tryCatch(tri(r), error = function(e) NA_real_))
write.table(out, "out_eulerr.txt", row.names = FALSE, col.names = FALSE)
cat("eulerr version", as.character(packageVersion("eulerr")), "\n")
sp <- list()
for (nm in c("configs", "traj_jupiter_earth", "traj_large_moon")) {
  Y <- if (nm == "configs") X else {
    j <- jsonlite_fallback <- NULL
    as.matrix(read.csv(paste0(nm, ".csv"), header = FALSE)) }
  Y <- Y[seq_len(min(nrow(Y), 40000)), ]
  noop <- function(r) r[7]
  t_noop <- min(replicate(3, system.time(apply(Y, 1, noop))[["elapsed"]]))
  t_call <- min(replicate(3, system.time(apply(Y, 1, tri))[["elapsed"]]))
  sp[[nm]] <- c(total_ns = 1e9 * t_call / nrow(Y), loop_overhead_ns = 1e9 * t_noop / nrow(Y))
}
print(sp)
saveRDS(sp, "speed_eulerr.rds")
write.csv(do.call(rbind, sp), "speed_eulerr.csv")
