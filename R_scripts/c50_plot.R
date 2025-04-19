library(C50)
library(cobalt)

feature_name <- "antithorombotic_therapy_detail_3.csv"
df <- read.csv(paste("../data/significant_action_features/",feature_name,sep=""))
df$aneurysm_site <- as.factor(df$aneurysm_site)
df$gender <- as.factor(df$gender)

df$outperform <- as.factor(df$outperform)

cov_names <- c("symptom","SAH_severity","aneurysm_shape","age","gender",
               "mRS_before","max_diameter","is_scheduled","aneurysm_site")
model_formula <- f.build("outperform",cov_names)

# cf <- 0.05
cf <- 0.1
model <- C5.0(formula=outperform~symptom+SAH_severity+aneurysm_shape+age+gender+
                  mRS_before+max_diameter+is_scheduled+aneurysm_site,
              data=df,rules=FALSE,control=C5.0Control(CF=cf))
plot(model)
summary(model)
