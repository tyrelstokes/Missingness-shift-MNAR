select_dag_function <- function(num){
  fname <- paste0("vae/parameter_presets/",
                  paste0("dag_",
                         paste0(num,"_parameters.R")))
  
  source(here::here(fname))
  
}