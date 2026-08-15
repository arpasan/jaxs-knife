# prior predictive before sampling
prior_predictive = True
fit = model.sample()
fit.idata.to_netcdf("inference_data.nc")
