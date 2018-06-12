
# Please change the working directory to the parent folder before running

#### enviroment setup ####

require(zoo)
require(dplyr)
require(xgboost)

#### function defination ####
read_data = function(filename){
  
  # The function is used to read raw data and aggregate by hour
  # Parameter: filename -- path of data
  # Return: dataframe of hourly data

  df = read.csv(filename,stringsAsFactors = FALSE)
  df$Timestamp = strptime(df$Timestamp,format = '%Y-%m-%d %H:%M:%S')
  # truncate timestamp
  df$Timestamp = paste(substr(df$Timestamp,1,13),':00:00',sep = '')
  
  new_df = summarise(group_by(df,meter_id,Timestamp,is_abnormal),
                     Values = mean(Values,na.rm = TRUE))
  return(new_df)
}
read_weather = function(filename_weather){
  # The function is used to read weather data and aggregate by hour
  # Parameter: filename -- path of data
  # Return: dataframe of hourly data
  weather = read.csv(filename_weather,stringsAsFactors = FALSE)
  weather$hour = paste(substr(weather$Timestamp,1,13),':00:00',sep = '')
  
  new_weather = summarise(group_by(weather,hour,Distance,site_id),
                          Temperature = mean(Temperature,na.rm = TRUE))
  colnames(new_weather)[1] = 'Timestamp'
  
  return(new_weather)
}

extract_features = function(df,weather){
  
  # The function is used to extract features from input_data
  # The features include:'month', 'week', 'hour', 'dayofyear','dayofmonth',
  #                     'dayofweek', 'temperature', 'is_off'
  # Parameter: df -- input data of current meter
  # Parameter: weather -- input weather data of current meter
  # Return: dataframe with features stated above
  
  df$month = (as.POSIXlt(df$Timestamp))$mon
  df$week = ceiling(((as.POSIXlt(df$Timestamp))$yday)/7)
  df$hour = (as.POSIXlt(df$Timestamp))$hour
  df$dayofyear = (as.POSIXlt(df$Timestamp))$yday
  df$dayofmonth = (as.POSIXlt(df$Timestamp))$mday
  df$dayofweek = (as.POSIXlt(df$Timestamp))$wday
  
  tmp_weather = subset(weather,select = c('Timestamp','Temperature'))
  df = merge(df,tmp_weather,by = 'Timestamp',all.x = TRUE)
  df$is_off = (df$dayofweek == 0)|(df$dayofweek == 6)
#  df$is_pre_off = df$dayofweek <= 1 
#  df$is_next_off = df$dayofweek >= 5
  
  return(df)
}
handle_holidays = function(df,holidays){
  # The function is used to add holiday information,and will change 
  # is_off,is_pre_off,is_next_off is the corresponding day is a public holiday
  # param: df -- site data after feature extraction
  # param: holidays -- site public holidays
  # return: df -- modified site data
  if(length(nrow(holidays)) == 0){
    return(df)
  }
  for(i in 1:nrow(holidays)){
    tmp_date = as.Date(holidays$Date[i])
    df$is_off[which(as.Date(df$Timestamp) == tmp_date)] = 1
  }
  return(df)
}
holiday_correction = function(df){
  # The function label days with abnormal low value with holiday label(is_off = 1).
  # Also it add two features is_pre_off & is_next_off, which indicates whether the day before/after
  # is a "holiday/low-value day"
  # param: df -- site_data after feature_extraction
  # return: df -- modifed site_data
  tmp1 = summarise(group_by(df,is_off),mean = mean(Values),std = sd(Values))
  tmp2 = summarise(group_by(df,as.Date(df$Timestamp),is_off),mean = mean(Values))
  day_low = tmp2$`as.Date(df$Timestamp)`[which(tmp2$is_off == 0 & tmp2$mean <= 
                                                 (tmp1$mean[2] + tmp1$std[2]))]
  for(i in 1:length(day_low)){
    tmp_date = as.Date(day_low[i])
    df$is_off[which(as.Date(df$Timestamp) == tmp_date)] = 1
  }
  
  df$date = as.Date(df$Timestamp)
  tmp3 = summarise(group_by(df,date,is_off))
  tmp3 = tmp3[order(tmp3$date),]
  
  tmp3$is_pre_off = c(0,tmp3$is_off[1:(nrow(tmp3)-1)])
  tmp3$is_next_off = c(tmp3$is_off[2:nrow(tmp3)],0)
  tmp4 = subset(tmp3,select = -is_off)
  
  df = merge(df,tmp4, by = 'date',all.x = TRUE)
  return(df)
}

xgb_model = function(df,n_fold){
  # The function is used to perform "n-fold" xgboost.Please refer to the report for details.
  # param: df -- site data
  # param: n_fold -- the number of cycles
  # return: dataframe with prediction value(average and details), true value and average error.
  
  features = c('month', 
               #'week',
               'hour', 'dayofyear',
              'dayofmonth', 'dayofweek', 'Temperature', 'is_off', 'is_pre_off',
              'is_next_off')
  X = data.matrix(subset(df,select = features))
  Y = data.matrix(df$Values)
  cut_number = cut(seq_len(nrow(X)),breaks = n_fold,labels = FALSE)
  
  result = data.frame(Timestamp = df$Timestamp,value_true = Y)
  for(i in 1:n_fold){
    X_train = X[-which(cut_number == i),]
    X_test = X[which(cut_number == i),]
    Y_train = Y[-which(cut_number == i)]
    Y_test = Y[which(cut_number == i)]
    
    dtrain = xgb.DMatrix(data = X_train,label = Y_train)
    dtest = xgb.DMatrix(data = X_test,label = Y_test)
    
    watchlist = list(train = dtrain,test = dtest)
    bst = xgb.train(data = dtrain,nround = 10000,watchlist = watchlist,
                    early_stopping_rounds = 100,
                    objective = 'reg:linear')
    pre = predict(bst,X)
    
    result[,paste('pre_',i,sep = '')] = pre
  }
  
  result$pre_ave = apply(result[,3:(2+n_fold)],1,mean)
  result$err = result$value_true - result$pre_ave
  
  return(result)
}
get_anomalies = function(xgb_result,n_sigma){
  # This function is used to select anomalies from xgboost result
  # param: xgb_result -- xgboost result from xgb_model function
  # param: n_sigma -- use n_sigma rule to select anomalies
  # return: return a list containing abnormal days & xgboost result by day
  xgb_result$date = as.Date(xgb_result$Timestamp)
  result_day = summarise(group_by(xgb_result,date),
                         true = mean(value_true),pre = mean(pre_ave),
                         err = mean(err))
  err_mean = mean(result_day$err)
  err_std = sd(result_day$err)
  anomalies = result_day$date[which(result_day$err >= err_mean+n_sigma*err_std)]
  
  rst_list = list(result_day,anomalies)
  return(rst_list)
}
pipeline = function(df,weather,holidays,n_fold,n_sigma){
  
  print('Extract features')
  df = extract_features(df,weather)
  df = handle_holidays(df,holidays)
  df = holiday_correction(df)
  df = df[-which(is.na(df$Temperature)),]
  print('Use xgb to train model')
  xgb = xgb_model(df,n_fold)
  print('Select out anomalies')
  rst = get_anomalies(xgb,n_sigma)
  return(rst)
}

#### main code ####

# read in holidays & weather
holidays = read.csv('data/raw/holidays.csv',stringsAsFactors = FALSE)
weather = read_weather('data/raw/weather.csv')

# anomalies site 234
print("Working on site 234 ... ")
weather_234 = weather[which((weather$site_id == '234_203') 
                            & (weather$Distance >10) & (weather$Distance < 13)),]
holidays_234 = FALSE
df_234 = read_data('data/processed/234_203.csv')
df_234 = df_234[which(as.Date(df_234$Timestamp) <= as.Date('2015-06-20')),]
rst_234 = pipeline(df_234,weather_234,holidays_234,4,3)
anomalies_234 = data.frame(meter_id = '234_203',date = rst_234[[2]])

# anomalies site 334
print("Working on site 334 ... ")
weather_334 = weather[which((weather$site_id == '334_61') & (weather$Distance >= 19)),]
holidays_334 = holidays[which(holidays$site_id == '334_61'),]
df_334 = read_data('data/processed/334_61.csv')
rst_334 = pipeline(df_334,weather_334,holidays_334,4,4)
anomalies_334 = data.frame(meter_id = '334_61',date = rst_334[[2]])
print(anomalies_334)

# anomalies site 38
print("Working on site 38 ... ")
weather_38 = weather[which((weather$site_id == '38')),]
holidays_38 = holidays[which(holidays$site_id == '038'),]
df_38 = read_data('data/processed/38_9687.csv')

df_38_1 = df_38[which(as.Date(df_38$Timestamp) >= as.Date('2012-12-21')),]
rst_38_1 = pipeline(df_38_1,weather_38,holidays_38,4,3.5)
df_38_2 = df_38[which(as.Date(df_38$Timestamp) <= as.Date('2012-12-21')),]
rst_38_2 = pipeline(df_38_2,weather_38,holidays_38,4,3.5)

anomalies_38 = rbind(data.frame(meter_id = '38_9686',date = rst_38_1[[2]]),
                     data.frame(meter_id = '38_9686',date = rst_38_2[[2]]))

# write result
rst_weekday = rbind(rbind(anomalies_234,anomalies_334),anomalies_38)

raw_234 = read.csv('data/processed/234_203.csv',stringsAsFactors = FALSE)
raw_334 = read.csv('data/processed/334_61.csv',stringsAsFactors = FALSE)
raw_38 = read.csv('data/processed/38_9687.csv',stringsAsFactors = FALSE)
raw = rbind(rbind(raw_234,raw_334),raw_38)

for(i in 1:nrow(rst_weekday)){
  raw[which((raw$meter_id == rst_weekday$meter_id[i]) & (substr(raw$Timestamp,1,10) == rst_weekday$date[i])),
      'is_abnormal'] = 'True'
}
print(length(which(raw$is_abnormal == 'True')))
write.csv(raw[,1:4],'data/result/res_weekday_r.csv',row.names = FALSE,quote = FALSE)

