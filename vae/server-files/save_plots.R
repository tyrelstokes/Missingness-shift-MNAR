


rmse_y_plot <- rmse_plot_gen(rmse_frame = rmse_y_df,
                             type = "target",
                             split_by_data = FALSE,
                             ipsum = FALSE,
                             mod_order = c("Z model",
                                           "Joint Model",
                                           "No outcome model",
                                           "No missingness model",
                                           "pvae",
                                           "notmiwae",
                                           "mice"))


rmse_x_plot <- rmse_plot_gen(rmse_frame = rmse_x_df,
                             type = "none",
                             split_by_data = TRUE,
                             ipsum = FALSE,
                             mod_order = c("Z model",
                                           "Joint Model",
                                           "No outcome model",
                                           "No missingness model",
                                           "pvae",
                                           "notmiwae",
                                           "mice"),
                             filter_out = TRUE)

save_plot(gplot = rmse_y_plot,
          fname = "rmse_y_plot")
save_plot(gplot = rmse_x_plot,
          fname = "rmse_x_plot")