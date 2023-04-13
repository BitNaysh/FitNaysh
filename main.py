from detect import fitNaysh
fit=fitNaysh()

fit.squat_counter()

if not fit.isCamOpen():
    fit.bicep_curl_counter()

if not fit.isCamOpen():
    fit.deadlift_counter()