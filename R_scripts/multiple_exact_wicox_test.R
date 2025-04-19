library(tidyverse)
library(hash)
library(glue)
library(coin)

num_states <- 83

for (i in seq(0,num_states-1)){
    df <- read_csv(glue("../data/action_features_mRS_diff/state_{i}.csv"),
                   show_col_types = FALSE)
    original_cols <- names(df)
    valid_cols <- original_cols[! original_cols=="mRS_diff"]
    
    p_values <- data.frame(rep(NaN,length(valid_cols)),row.names = valid_cols)
    colnames(p_values) <- "p_value"
    for (col in valid_cols){
        col_df <- df[,col]
        outcomes <- data.frame(
            col_df,
            df["mRS_diff"]
        )
        colnames(outcomes) <- c("treated","mRS_diff")
        outcomes$treated <- as.factor(outcomes$treated)
        if (length(unique(df[[col]]))==2 & length(unique(outcomes$mRS_diff))>1){
            result <- wilcox_test(mRS_diff~treated,data=outcomes,distribution="exact",alternative="two.sided")
            p_values[col,1] <- pvalue(result)
        }
    }
    valid_p <- na.omit(p_values)
    adjusted_p <- p.adjust(unlist(valid_p),method="fdr")
    valid_p[,1] <- as.numeric(adjusted_p)
    
    write.csv(valid_p,glue("../data/action_features_mRS_diff/wilcox/state_{i}.csv"),row.names = TRUE)
    if (i%%10==0){
        print(glue("{i} done"))
    }
    
}
