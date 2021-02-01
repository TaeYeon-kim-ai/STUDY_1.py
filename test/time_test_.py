import datetime

start = datetime.datetime.now()
model.fit(x_train,y_train)
end = datetime.datetime.now()
print("time", end-start)

#n_job = -1 time 0:00:00.029920
#n_job = 8 time 0:00:00.035905
#n_job = 1 time 0:00:00.030917
#n_job = 2 time 0:00:00.038897
#n_job = 3 time 0:00:00.029921
#n_job = 4 time 0:00:00.037899
#n_job = 5 time 0:00:00.036902
#n_job = 6 time 0:00:00.034907
#n_job = 7 time 0:00:00.030917