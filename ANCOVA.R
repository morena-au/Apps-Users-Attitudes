# Rmcorr and ANCOVA
library(rmcorr)
library(dplyr)
library(readxl)

# Import datasets
setwd("~/Projects/AppInvestigation/Datasets")
group_4_raw <- read.csv('group_4.csv')
group_3_raw <- read.csv('group_3.csv')
group_2_raw <- read.csv('group_2.csv')
group_1_raw <- read.csv('group_1.csv')
Apps <- read.csv("Apps.csv")
personality <- read_excel("Part_Personality.xlsx")

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

rm(group_1_raw, group_2_raw, group_3_raw, group_4_raw)

# Add all the other columns

colnames(Apps)[1] = 'Participant'

#Datum[1,2,3,4]
group_1 = merge(group_1, Apps[, c('Participant', 'Datum_1_Interview')], by ='Participant')
colnames(group_1)[7] = 'Datum'
group_1[, 'V17'] <- NA
group_1[, 'V18'] <- NA
group_1[, 'V19'] <- NA
group_1[, 'V20'] <- NA
group_1[, 'V21'] <- NA

# GROUP 1
# subset by number of interview
group_2_1 = subset(group_2, Trial==1)
group_2_1 = merge(group_2_1, Apps[, c('Participant', 'Datum_1_Interview')], by ='Participant')
colnames(group_2_1)[7] = 'Datum'
group_2_1[, 'V17'] <- NA
group_2_1[, 'V18'] <- NA
group_2_1[, 'V19'] <- NA
group_2_1[, 'V20'] <- NA
group_2_1[, 'V21'] <- NA


# V17[2,3,4]
# #V18[2,3,4], V19[2,3,4], V20[2,3,4], V21[2,3,4]
group_2_2 = subset(group_2, Trial == 2)
group_2_2 = merge(group_2_2, Apps[, c('Participant', 'Datum_2_Interview', 
                                      'V17_2_Interview', 'V18_2_Interview',
                                      'V19_2_Interview', 'V20_2_Interview',
                                      'V21_2_Interview')], by ='Participant')
colnames(group_2_2)[7] = 'Datum'
colnames(group_2_2)[8] = 'V17'
colnames(group_2_2)[9] = 'V18'
colnames(group_2_2)[10] = 'V19'
colnames(group_2_2)[11] = 'V20'
colnames(group_2_2)[12] = 'V21'


group_2 <- rbind(group_2_1, group_2_2)
rm(group_2_1, group_2_2)

# GROUP 3
# subset by number of interview
group_3_1 = subset(group_3, Trial==1)
group_3_1 = merge(group_3_1, Apps[, c('Participant', 'Datum_1_Interview')], by ='Participant')
colnames(group_3_1)[7] = 'Datum'
group_3_1[, 'V17'] <- NA
group_3_1[, 'V18'] <- NA
group_3_1[, 'V19'] <- NA
group_3_1[, 'V20'] <- NA
group_3_1[, 'V21'] <- NA

group_3_2 = subset(group_3, Trial == 2)
group_3_2 = merge(group_3_2, Apps[, c('Participant', 'Datum_2_Interview', 
                                      'V17_2_Interview', 'V18_2_Interview',
                                      'V19_2_Interview', 'V20_2_Interview',
                                      'V21_2_Interview')], by ='Participant')
colnames(group_3_2)[7] = 'Datum'
colnames(group_3_2)[8] = 'V17'
colnames(group_3_2)[9] = 'V18'
colnames(group_3_2)[10] = 'V19'
colnames(group_3_2)[11] = 'V20'
colnames(group_3_2)[12] = 'V21'

group_3_3 = subset(group_3, Trial == 3)
group_3_3 = merge(group_3_3, Apps[, c('Participant', 'Datum_3_Interview', 
                                      'V17_3_Interview', 'V18_3_Interview',
                                      'V19_3_Interview', 'V20_3_Interview',
                                      'V21_3_Interview')], by ='Participant')
colnames(group_3_3)[7] = 'Datum'
colnames(group_3_3)[8] = 'V17'
colnames(group_3_3)[9] = 'V18'
colnames(group_3_3)[10] = 'V19'
colnames(group_3_3)[11] = 'V20'
colnames(group_3_3)[12] = 'V21'

group_3 <- rbind(group_3_1, group_3_2, group_3_3)
rm(group_3_1, group_3_2, group_3_3)

# GROUP 4
# subset by number of interview
group_4_1 = subset(group_4, Trial==1)
group_4_1 = merge(group_4_1, Apps[, c('Participant', 'Datum_1_Interview')], by ='Participant')
colnames(group_4_1)[7] = 'Datum'
group_4_1[, 'V17'] <- NA
group_4_1[, 'V18'] <- NA
group_4_1[, 'V19'] <- NA
group_4_1[, 'V20'] <- NA
group_4_1[, 'V21'] <- NA

group_4_2 = subset(group_4, Trial == 2)
group_4_2 = merge(group_4_2, Apps[, c('Participant', 'Datum_2_Interview', 
                                      'V17_2_Interview', 'V18_2_Interview',
                                      'V19_2_Interview', 'V20_2_Interview',
                                      'V21_2_Interview')], by ='Participant')
colnames(group_4_2)[7] = 'Datum'
colnames(group_4_2)[8] = 'V17'
colnames(group_4_2)[9] = 'V18'
colnames(group_4_2)[10] = 'V19'
colnames(group_4_2)[11] = 'V20'
colnames(group_4_2)[12] = 'V21'

group_4_3 = subset(group_4, Trial == 3)
group_4_3 = merge(group_4_3, Apps[, c('Participant', 'Datum_3_Interview', 
                                      'V17_3_Interview', 'V18_3_Interview',
                                      'V19_3_Interview', 'V20_3_Interview',
                                      'V21_3_Interview')], by ='Participant')
colnames(group_4_3)[7] = 'Datum'
colnames(group_4_3)[8] = 'V17'
colnames(group_4_3)[9] = 'V18'
colnames(group_4_3)[10] = 'V19'
colnames(group_4_3)[11] = 'V20'
colnames(group_4_3)[12] = 'V21'

group_4_4 = subset(group_4, Trial == 4)
group_4_4 = merge(group_4_4, Apps[, c('Participant', 'Datum_4_Interview', 
                                      'V17_4_Interview', 'V18_4_Interview',
                                      'V19_4_Interview', 'V20_4_Interview',
                                      'V21_4_Interview')], by ='Participant')
colnames(group_4_4)[7] = 'Datum'
colnames(group_4_4)[8] = 'V17'
colnames(group_4_4)[9] = 'V18'
colnames(group_4_4)[10] = 'V19'
colnames(group_4_4)[11] = 'V20'
colnames(group_4_4)[12] = 'V21'

group_4 <- rbind(group_4_1, group_4_2, group_4_3, group_4_4)
rm(group_4_1, group_4_2, group_4_3, group_4_4)

# V1, V01, V2, V3, V10, V11, V13, V14
time_row <- rbind(group_1, group_2, group_3, group_4)

time_row = merge(time_row, Apps[, c("Participant", "V1", "V01", "V2", "V3",
                         "V10", "V11", "V13", "V14")], by = 'Participant')

# Merge with personality
# create identifier


time_row[,'Probanden_ID__lfdn'] <- gsub('__[^__]+$', '', time_row$Participant)

personality['Probanden_ID__lfdn'] <- paste0(personality$Probanden_ID, '__', personality$lfdn)

time_row = merge(time_row, personality[, c("Probanden_ID__lfdn", "Gender", "Age", "HABIT", "NovSeek",
                                        "PrivConc")], by = 'Probanden_ID__lfdn')

write.csv(time_row, file ='~/Projects/AppInvestigation/Datasets/time_row.csv', row.names=FALSE)


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

