

small_x <- rmse_x_comb %>%
  dplyr::filter(type == "combined") %>%
  dplyr::mutate(model_order = forcats::fct_reorder(model_name,-rmse))

small_y <- rmse_y_comb %>%
  dplyr::mutate(model_order = forcats::fct_reorder(model_name,-rmse))


upp <- 3.0
low <- 0.5


plotter <- small_x %>%
  ggplot2::ggplot(ggplot2::aes(x = model_order, y= rmse)) +
  ggplot2::geom_boxplot(ggplot2::aes(fill = model_name),outlier.shape = NA) + 
  #ggplot2::facet_wrap(~data_type) +
  ggplot2::guides(fill="none") +
  ggplot2::ylab("RMSE")+
  ggplot2::xlab("")+
  hrbrthemes::theme_ipsum()+
  ggplot2::coord_flip(ylim = c(low,upp))


plotter




plotter_dag <- small_x %>%
  ggplot2::ggplot(ggplot2::aes(x = model_order, y= rmse)) +
  ggplot2::geom_boxplot(ggplot2::aes(fill = model_name),outlier.shape = NA) + 
  ggplot2::coord_flip(ylim = c(low,upp))+
  ggplot2::facet_wrap(~dag_type)+
  ggplot2::guides(fill="none") +
  ggplot2::ylab("RMSE")+
  ggplot2::xlab("")+
  hrbrthemes::theme_ipsum()


plotter_dag




plotter_dag1 <- small_x %>%
  dplyr::filter(dag_type == 1) %>%
  ggplot2::ggplot(ggplot2::aes(x = model_order, y= rmse)) +
  ggplot2::geom_boxplot(ggplot2::aes(fill = model_name),outlier.shape = NA) + 
  ggplot2::coord_flip(ylim = c(low,upp))+
  ggplot2::guides(fill="none") +
  ggplot2::ylab("RMSE")+
  ggplot2::xlab("")+
  hrbrthemes::theme_ipsum()

plotter_dag1


plotter_dag2 <- small_x %>%
  dplyr::filter(dag_type == 2) %>%
  ggplot2::ggplot(ggplot2::aes(x = model_order, y= rmse)) +
  ggplot2::geom_boxplot(ggplot2::aes(fill = model_name),outlier.shape = NA) + 
  ggplot2::coord_flip(ylim = c(low,upp))+
  ggplot2::guides(fill="none") +
  ggplot2::ylab("RMSE")+
  ggplot2::xlab("")+
  hrbrthemes::theme_ipsum()

plotter_dag2



plotter_dag3 <- small_x %>%
  dplyr::filter(dag_type == 3) %>%
  ggplot2::ggplot(ggplot2::aes(x = model_order, y= rmse)) +
  ggplot2::geom_boxplot(ggplot2::aes(fill = model_name),outlier.shape = NA) + 
  ggplot2::coord_flip(ylim = c(low,upp))+
  ggplot2::guides(fill="none") +
  ggplot2::ylab("RMSE")+
  ggplot2::xlab("")+
  hrbrthemes::theme_ipsum()

plotter_dag3


plotter_dag4 <- small_x %>%
  dplyr::filter(dag_type == 4) %>%
  ggplot2::ggplot(ggplot2::aes(x = model_order, y= rmse)) +
  ggplot2::geom_boxplot(ggplot2::aes(fill = model_name),outlier.shape = NA) + 
  ggplot2::coord_flip(ylim = c(low,upp))+
  ggplot2::guides(fill="none") +
  ggplot2::ylab("RMSE")+
  ggplot2::xlab("")+
  hrbrthemes::theme_ipsum()

plotter_dag4


# Only using super non-linear runs



plotter_nl <- small_x %>%
 # dplyr::filter(n_layers_x == 4) %>%
  ggplot2::ggplot(ggplot2::aes(x = model_order, y= rmse)) +
  ggplot2::geom_boxplot(ggplot2::aes(fill = model_name),outlier.shape = NA) + 
  ggplot2::coord_flip(ylim = c(low,upp))+
  ggplot2::guides(fill="none") +
  ggplot2::ylab("RMSE")+
  ggplot2::xlab("")+
  hrbrthemes::theme_ipsum()

plotter_nl




plotter_dag4 <- small_x %>%
  #dplyr::filter(dag_type %in% c(2)) %>%
  dplyr::filter(n == 500) %>%
  ggplot2::ggplot(ggplot2::aes(x = model_order, y= rmse)) +
  ggplot2::geom_boxplot(ggplot2::aes(fill = model_name),outlier.shape = NA) + 
  ggplot2::coord_flip(ylim = c(low,upp))+
  ggplot2::guides(fill="none") +
  ggplot2::ylab("RMSE")+
  ggplot2::xlab("")+
  hrbrthemes::theme_ipsum()

plotter_dag4
