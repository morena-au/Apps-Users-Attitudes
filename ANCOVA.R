# Rmcorr and ANCOVA
library(rmcorr)
library(dplyr)


# Import datasets
setwd("~/Projects/AppInvestigation/Datasets")
group_4_raw <- read.csv('group_4.csv')
group_3_raw <- read.csv('group_3.csv')
group_2_raw <- read.csv('group_2.csv')
group_1_raw <- read.csv('group_1.csv')


# Data format for the analysis
# Drop unused columns
group_4_raw <- select(group_4_raw, -c("Probanden_ID__lfdn", "V2", "V17_2_Interview", "V17_3_Interview",
                                      "V17_4_Interview", "V18_2_Interview", "V18_3_Interview", "V18_4_Interview"))
group_3_raw <- select(group_3_raw, -c("Probanden_ID__lfdn", "V2", "V17_2_Interview", "V17_3_Interview",
                                      "V17_4_Interview", "V18_2_Interview", "V18_3_Interview", "V18_4_Interview"))
group_2_raw <- select(group_2_raw, -c("Probanden_ID__lfdn", "V2", "V17_2_Interview", "V17_3_Interview",
                                      "V17_4_Interview", "V18_2_Interview", "V18_3_Interview", "V18_4_Interview"))
group_1_raw <- select(group_1_raw, -c("Probanden_ID__lfdn", "V2", "V17_2_Interview", "V17_3_Interview",
                                      "V17_4_Interview", "V18_2_Interview", "V18_3_Interview", "V18_4_Interview"))


# replicate all participants 4 times
Participant <- group_4_raw["Probanden_ID__lfdn__AppNr"]
Participant <- Participant[rep(seq_len(nrow(Participant)), 4), ]

# Insert Interview's number
Trial <- c(rep(1, nrow(group_4_raw)), rep(2, nrow(group_4_raw)), 
           rep(3, nrow(group_4_raw)), rep(4, nrow(group_4_raw)))

# Append variable from each interview one after another
V4 <- c(group_4_raw["V4_1_Interview"][,], group_4_raw["V4_2_Interview"][,],
        group_4_raw["V4_3_Interview"][,], group_4_raw["V4_4_Interview"][,])

V6 <- c(group_4_raw["V6_1_Interview"][,], group_4_raw["V6_2_Interview"][,],
        group_4_raw["V6_3_Interview"][,], group_4_raw["V6_4_Interview"][,])

V12 <- c(group_4_raw["V12_1_Interview"][,], group_4_raw["V12_2_Interview"][,],
         group_4_raw["V12_3_Interview"][,], group_4_raw["V12_4_Interview"][,])

# group 4 final dataset
group_4 <- data.frame(Participant, Trial, V4, V6, V12)
group_4['Group'] = 4

# replicate all participants 3 times
Participant <- group_3_raw["Probanden_ID__lfdn__AppNr"]
Participant <- Participant[rep(seq_len(nrow(Participant)), 3), ]

# Insert Interview's number
Trial <- c(rep(1, nrow(group_3_raw)), rep(2, nrow(group_3_raw)), 
           rep(3, nrow(group_3_raw)))

# Append variable from each interview one after another
V4 <- c(group_3_raw["V4_1_Interview"][,], group_3_raw["V4_2_Interview"][,],
        group_3_raw["V4_3_Interview"][,])

V6 <- c(group_3_raw["V6_1_Interview"][,], group_3_raw["V6_2_Interview"][,],
        group_3_raw["V6_3_Interview"][,])

V12 <- c(group_3_raw["V12_1_Interview"][,], group_3_raw["V12_2_Interview"][,],
         group_3_raw["V12_3_Interview"][,])

# group 3 final dataset
group_3 <- data.frame(Participant, Trial, V4, V6, V12)
group_3['Group'] = 3

# replicate all participants 2 times
Participant <- group_2_raw["Probanden_ID__lfdn__AppNr"]
Participant <- Participant[rep(seq_len(nrow(Participant)), 2), ]

# Insert Interview's number
Trial <- c(rep(1, nrow(group_2_raw)), rep(2, nrow(group_2_raw)))

# Append variable from each interview one after another
V4 <- c(group_2_raw["V4_1_Interview"][,], group_2_raw["V4_2_Interview"][,])

V6 <- c(group_2_raw["V6_1_Interview"][,], group_2_raw["V6_2_Interview"][,])

V12 <- c(group_2_raw["V12_1_Interview"][,], group_2_raw["V12_2_Interview"][,])

# group 2 final dataset
group_2 <- data.frame(Participant, Trial, V4, V6, V12)
group_2['Group'] = 2

# replicate all participants 1 times
Participant <- group_1_raw["Probanden_ID__lfdn__AppNr"]
Participant <- Participant[rep(seq_len(nrow(Participant)), 1), ]

# Insert Interview's number
Trial <- c(rep(1, nrow(group_1_raw)))

# Append variable from each interview one after another
V4 <- c(group_1_raw["V4_1_Interview"][,])

V6 <- c(group_1_raw["V6_1_Interview"][,])

V12 <- c(group_1_raw["V12_1_Interview"][,])

# group 1 final dataset
group_1 <- data.frame(Participant, Trial, V4, V6, V12)
group_1['Group'] = 1

time_row <- rbind(group_1, group_2, group_3, group_4)

write.csv(time_row, file ='~/Projects/AppInvestigation/Datasets/time_row.csv')


# RMCORR: assesses the intra-individual or longitudinal change
# To be able to plot sample randomly roughly 10 people from the dataset

# set seed to make the sampling reproducible
set.seed(123)
group_4_ind <- sample(seq_len(nrow(group_4)), size = 10)
group_4_sample <- group_4[group_4$Participant %in% group_4[group_4_ind,1],]

V6_V4_sample.rmc <- rmcorr(Participant, V6, V4, group_4_sample)
print(V6_V4_sample.rmc)
plot(V6_V4_sample.rmc, overall = F, lty=2, xlab = 'V6', ylab = 'V4')

# Overall rmcorr output
rmc.out <- rmcorr(Participant, V4, V6, group_4)
print(rmc.out)


# COMMENTS:
# Small effect sizes for rmcorr (r = -0.15) may be caused by heterogenous slopes
# (poor model fit), or by consistently near-zero slopes across subjects, or by
# restriction in the range of one or both measures.

