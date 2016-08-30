require(pacman)

p_load(foreach,
       parallel,
       mcparallelDo,
       plyr,
       dplyr,
       tidyr,
       tibble,
       caret,
       caretEnsemble,
       xgboost)

load('descr.RData')
load('mutagen.RData')

# clusterSetRNGStream(1729)

set.seed(1)

inTrain <- createDataPartition(mutagen, p = 3 / 4, list = FALSE)
trainDescr <- descr[inTrain,]
testDescr <- descr[-inTrain,]
trainClass <- mutagen[inTrain]
testClass <- mutagen[-inTrain]

# mcparallelDo({
#   nearZeroVar(
#     trainDescr,
#     freqCut = 19 / 5,
#     uniqueCut = 20,
#     saveMetrics = FALSE
#   )
# },
# targetValue = 'nzvars_pid', verbose = FALSE)
# nzvars <- mccollect()
# nzvars <- nzvars %>%
#   unlist() %>% as.vector()
# gc()
# rm(nzvars_pid)



nzvars <-
  nearZeroVar(trainDescr, freqCut = 19/5,
              uniqueCut = 20, saveMetrics = FALSE)

descrNZ <- descr[, -nzvars]
trainDescrNZ <- trainDescr[, -nzvars]
testDescrNZ <- testDescr[, -nzvars]


descrCorr <- cor(descrNZ)
highCorr <- findCorrelation(descrCorr, 0.90)

descrNZ <- descrNZ[, -highCorr]
trainDescrNZ <- trainDescrNZ[, -highCorr]
testDescrNZ <- testDescrNZ[, -highCorr]

xgbTrCtrl <- trainControl(method = 'cv', number = 5,
                          #repeats = 5,
                          classProbs = TRUE, search = 'grid',
                          allowParallel = TRUE)

tg <- expand.grid(gamma = 0, min_child_weight = c(.1, .5, 1, 3),
                  max_depth = c(2, 5, 25), eta = .3, colsample_bytree = 1,
                  nrounds = c(10, 100, 1000))

xgb1 <- train(x = trainDescrNZ, y = trainClass,
              method = 'xgbTree', trControl = xgbTrCtrl, tuneGrid = tg)

#nzvars <- mccollect()
#nzvars <- nzvars %>%
#  unlist() %>% as.vector()
#gc()
#rm(nzvars_pid)
