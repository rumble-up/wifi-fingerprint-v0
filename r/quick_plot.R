library(esquisse)

df_x51 <- read.csv(file="/home/chief/0githubi/wifi-fingerprint/predictions/best_guesses/mvp_autotune_rand8_clean.csv")
df_all_train <- read.csv(file="/home/chief/0githubi/wifi-fingerprint/predictions/best_guesses/mvp_autotune_rand8_all_training.csv")

f0 <- read.csv(file="/home/chief/0githubi/wifi-fingerprint/predictions/RF_autotune_rand3.csv")
f0 <- read.csv(file="/home/chief/0githubi/wifi-fingerprint/predictions/RF_no_flr_pred_rand3.csv")
f0 <- read.csv(file="/home/chief/0githubi/wifi-fingerprint/predictions/LATITUDE_rand9_rf_rscv_all_train.csv")

esquisse::esquisser(f0)
