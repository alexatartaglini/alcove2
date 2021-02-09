library(tidyverse)
library(fs)
library(pracma)
library(ggthemes)

## combine all csv files into a single data frame
data_dir <- "results"
df <- data_dir %>% 
    dir_ls(regexp = "\\.csv$") %>%
    map_dfr(read_csv)
write_csv(df, "data/all_simulations.csv")

## read in all simulations again
df <- read_csv("data/all_simulations.csv")

## separate out alcove and mlp simulations
alcove_df <- df %>%
    filter(model_type == 'alcove')
write_csv(alcove_df, "data/alcove_simulations.csv")

mlp_df <- df %>%
    filter(model_type == 'mlp')
write_csv(mlp_df, "data/mlp_simulations.csv")

## check counts for each simulation
alcove_df %>%
    group_by(net_type, image_set) %>%
    summarise(n = n())
mlp_df %>%
    group_by(net_type, image_set) %>%
    summarise(n = n())

## alcove analyses
## averaging results across permutations
alcove_avg_df <- alcove_df %>%
    group_by(net_type, image_set, lr_association, lr_attention, c, phi, type, epoch) %>%
    summarise(prob_correct = mean(prob_correct))

## plot learning curves
## TODO: turn this into a function that does this for all net x image sets
net_image_df <- alcove_avg_df %>%
    filter(net_type == 'resnet50', image_set == 'shj_images_set3')

## rename columns to show up on the facet
net_image_df <- net_image_df %>%
    mutate(lr_association = paste0('lr_assoc: ', lr_association),
           lr_attention = paste0('lr_attn: ', lr_attention),
           c = paste0('c: ', c),
           phi = paste0('phi: ', phi))

ggplot(net_image_df, aes(x = epoch, y = prob_correct, color = factor(type))) +
    geom_line() +
    facet_grid(factor(c) + factor(phi) ~ factor(lr_association) + factor(lr_attention))

## calculate integrals
alcove_avg_integrals <- alcove_avg_df %>%
    group_by(net_type, image_set, lr_association, lr_attention, c, phi, type) %>%
    summarise(auc = trapz(epoch, prob_correct))

## calculate correct orders
correct_orders <- c("123456", "123546", "124356", "124536", "125346", "125436")
alcove_avg_orders <- alcove_avg_integrals %>%
    group_by(net_type, image_set, lr_association, lr_attention, c, phi) %>%
    arrange(desc(auc)) %>%
    summarise(order = toString(type)) %>%
    mutate(order = str_replace_all(order, ", ", "")) %>%
    mutate(correct = order %in% correct_orders)

## perform join
alcove_avg_df <- alcove_avg_df %>%
    left_join(alcove_avg_orders)

## plot correct learning curves
alcove_avg_correct_df <- alcove_avg_df %>%
    filter(correct)

ggplot(alcove_avg_correct_df, aes(x = epoch, y = prob_correct, color = factor(type))) +
    geom_line() +
    facet_wrap(~ c + phi + lr_association + lr_attention)

## calculate spearman rank correlation
correct_ranks <- c(1, 3, 3, 3, 5, 6)
alcove_avg_correlations <- alcove_avg_integrals %>%
    group_by(net_type, image_set, lr_association, lr_attention, c, phi) %>%
    arrange(auc) %>%
    mutate(order = 7 - type) %>%
    group_by(net_type, image_set, lr_association, lr_attention, c, phi) %>%
    summarise(correlation = cor(order, correct_ranks)) %>%
    mutate(correct = correlation >= 0.941,
           attention = lr_attention > 0)

alcove_avg_correlations %>%
    group_by() %>%
    summarise(prop_correct = sum(correct) / n()) %>%
    print.data.frame()

## density plot of correlation values
ggplot(alcove_avg_correlations, aes(x = correlation, group = attention, fill = attention)) +
    geom_density(size=0.1, adjust=1/4, alpha=0.5, position ="stack") +
    ## geom_histogram(binwidth = 0.05) +
    geom_vline(xintercept = 0.941, linetype='dashed') +
    facet_grid(image_set ~ attention + net_type) +
    xlim(0, 1) +
    theme(legend.position = 'bottom')

## keeping permutations distinct
alcove_all_df <- alcove_df %>%
    group_by(net_type, image_set, lr_association, lr_attention, c, phi, type, permutation, epoch) %>%
    summarise(prob_correct = mean(prob_correct))

## calculate integrals
alcove_all_integrals <- alcove_all_df %>%
    group_by(net_type, image_set, lr_association, lr_attention, c, phi, type, permutation) %>%
    summarise(auc = trapz(epoch, prob_correct))

## calculate correct orders
correct_orders <- c("123456", "123546", "124356", "124536", "125346", "125436")
alcove_all_orders <- alcove_all_integrals %>%
    group_by(net_type, image_set, lr_association, lr_attention, c, phi, permutation) %>%
    arrange(desc(auc)) %>%
    summarise(order = toString(type)) %>%
    mutate(order = str_replace_all(order, ", ", "")) %>%
    mutate(correct = order %in% correct_orders)

## calculate spearman rank correlation
correct_ranks <- c(1, 3, 3, 3, 5, 6)
alcove_all_correlations <- alcove_all_integrals %>%
    group_by(net_type, image_set, lr_association, lr_attention, c, phi, permutation) %>%
    arrange(auc) %>%
    mutate(order = 7 - type) %>%
    group_by(net_type, image_set, lr_association, lr_attention, c, phi, permutation) %>%
    summarise(correlation = cor(order, correct_ranks)) %>%
    mutate(correct = correlation >= 0.941,
           attention = lr_attention > 0)

alcove_all_correlations %>%
    group_by(attention) %>%
    summarise(prop_correct = sum(correct) / n()) %>%
    print.data.frame()

## density plot of correlation values
ggplot(alcove_all_correlations, aes(x = correlation, group = factor(permutation), fill = factor(permutation))) +
    ## geom_density(size=0.1, adjust=1/2, alpha=0.5, position ="stack") +
    geom_histogram(binwidth = 0.05) +
    geom_vline(xintercept = 0.941, linetype='dashed') +
    facet_grid(image_set ~ attention + net_type) +
    xlim(0, 1) +
    theme_few() +
    theme(legend.position = 'bottom')

## mlp analyses
## averaging results across permutations
mlp_avg_df <- mlp_df %>%
    group_by(net_type, image_set, lr_association, phi, type, epoch) %>%
    summarise(prob_correct = mean(prob_correct))

## plot learning curves
## TODO: turn this into a function that does this for all net x image sets
net_image_df <- mlp_avg_df

## rename columns to show up on the facet
net_image_df <- net_image_df %>%
    mutate(lr_association = paste0('lr_assoc: ', lr_association),
           phi = paste0('phi: ', phi))

ggplot(net_image_df, aes(x = epoch, y = prob_correct, color = factor(type))) +
    geom_line() +
    facet_grid(net_type + image_set ~ factor(lr_association) + factor(phi))

## calculate integrals
mlp_avg_integrals <- mlp_avg_df %>%
    group_by(net_type, image_set, lr_association, phi, type) %>%
    summarise(auc = trapz(epoch, prob_correct))

## calculate correct orders
correct_orders <- c("123456", "123546", "124356", "124536", "125346", "125436")
mlp_avg_orders <- mlp_avg_integrals %>%
    group_by(net_type, image_set, lr_association, phi) %>%
    arrange(desc(auc)) %>%
    summarise(order = toString(type)) %>%
    mutate(order = str_replace_all(order, ", ", "")) %>%
    mutate(correct = order %in% correct_orders)

## calculate spearman rank correlation
correct_ranks <- c(1, 3, 3, 3, 5, 6)
mlp_avg_correlations <- mlp_avg_integrals %>%
    group_by(net_type, image_set, lr_association, phi) %>%
    arrange(auc) %>%
    mutate(order = 7 - type) %>%
    group_by(net_type, image_set, lr_association, phi) %>%
    summarise(correlation = cor(order, correct_ranks)) %>%
    mutate(correct = correlation >= 0.941)

mlp_avg_correlations %>%
    group_by(net_type, image_set) %>%
    summarise(prop_correct = sum(correct) / n()) %>%
    print.data.frame()

## density plot of correlation values
ggplot(mlp_avg_correlations, aes(x = correlation)) +
    geom_density(size=0.1, adjust=1/4, alpha=0.5, position ="stack") +
    ## geom_histogram(binwidth = 0.05) +
    geom_vline(xintercept = 0.941, linetype='dashed') +
    facet_grid(image_set ~ net_type) +
    xlim(0, 1) +
    theme(legend.position = 'bottom')

## keeping each permutation separate
mlp_all_df <- mlp_df %>%
    group_by(net_type, image_set, lr_association, phi, type, permutation, epoch) %>%
    summarise(prob_correct = mean(prob_correct))

## ## plot learning curves
## ## TODO: turn this into a function that does this for all net x image sets
## net_image_df <- mlp_all_df

## ## rename columns to show up on the facet
## net_image_df <- net_image_df %>%
##     mutate(lr_association = paste0('lr_assoc: ', lr_association),
##            phi = paste0('phi: ', phi))

## ggplot(net_image_df, aes(x = epoch, y = prob_correct, color = factor(type))) +
##     geom_line() +
##     facet_grid(net_type + image_set ~ factor(lr_association) + factor(phi))

## calculate integrals
mlp_all_integrals <- mlp_all_df %>%
    group_by(net_type, image_set, lr_association, phi, permutation, type) %>%
    summarise(auc = trapz(epoch, prob_correct))

## calculate correct orders
correct_orders <- c("123456", "123546", "124356", "124536", "125346", "125436")
mlp_all_orders <- mlp_all_integrals %>%
    group_by(net_type, image_set, lr_association, phi, permutation) %>%
    arrange(desc(auc)) %>%
    summarise(order = toString(type)) %>%
    mutate(order = str_replace_all(order, ", ", "")) %>%
    mutate(correct = order %in% correct_orders)

## calculate spearman rank correlation
correct_ranks <- c(1, 3, 3, 3, 5, 6)
mlp_all_correlations <- mlp_all_integrals %>%
    group_by(net_type, image_set, lr_association, phi, permutation) %>%
    arrange(auc) %>%
    mutate(order = 7 - type) %>%
    group_by(net_type, image_set, lr_association, phi, permutation) %>%
    summarise(correlation = cor(order, correct_ranks)) %>%
    mutate(correct = correlation >= 0.941)

mlp_all_correlations %>%
    group_by(net_type, image_set) %>%
    summarise(prop_correct = sum(correct) / n()) %>%
    print.data.frame()

## density plot of correlation values
ggplot(mlp_all_correlations, aes(x = correlation)) +
    geom_density(size=0.1, adjust=1/4, alpha=0.5, position ="stack") +
    ## geom_histogram(binwidth = 0.05) +
    geom_vline(xintercept = 0.941, linetype='dashed') +
    facet_grid(image_set ~ net_type) +
    theme(legend.position = 'bottom')

## plot correct runs
mlp_all_correct_df <- mlp_all_df %>%
    left_join(mlp_all_correlations) %>%
    filter(correct)

ggplot(mlp_all_correct_df, aes(x = epoch, y = prob_correct, color = factor(type))) +
    geom_line() +
    facet_wrap(~ interaction(net_type, image_set, lr_association, phi, permutation)) +
    ggtitle("Successful MLP simulations (by individual permutations)") +
    theme_few()

## combine correlation plots for cogsci figures
alcove_all_correlations <- alcove_all_correlations %>%
    mutate(model_name = ifelse(attention, "CNN-ALCOVE-Attn", "CNN-ALCOVE-No-Attn"))

mlp_all_correlations <- mlp_all_correlations %>%
    mutate(model_name = "CNN-MLP")

all_correlations <- bind_rows(list(alcove_all_correlations, mlp_all_correlations))

## rename column values
all_correlations <- all_correlations %>%
    mutate(image_set = case_when(
               image_set == 'shj_images_set1' ~ "SHJ Set 1",
               image_set == 'shj_images_set2' ~ "SHJ Set 2",
               image_set == 'shj_images_set3' ~ "SHJ Set 3",
               TRUE ~ "other")) %>%
    mutate(net_type = case_when(
               net_type == 'vgg16' ~ "VGG-16",
               net_type == 'resnet18' ~ "ResNet-18",
               net_type == 'resnet50' ~ "ResNet-50",
               TRUE ~ "other"))

## normalize
ggplot(all_correlations, aes(x = correlation, group = factor(permutation), fill = factor(permutation))) +
    ## geom_density(size=0.1, adjust=1/2, alpha=0.5, position ="stack") +
    geom_histogram(binwidth = 0.1, boundary = 0, aes(y = stat(count / sum(count)))) +
    geom_vline(xintercept = 0.9, linetype='dashed') +
    facet_grid(model_name ~ image_set + net_type, scales="free_y") +
    xlab("Spearman Correlation") +
    ylab("Proportion") +
    theme_few() +
    scale_fill_discrete("Permutation") +
    theme(legend.position = "bottom",
        panel.border = element_rect(colour = "black", size = 1),
        axis.title.x = element_text(size = 20),
        axis.title.y = element_text(size = 20),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),
        axis.ticks.x = element_line(colour = "black"),
        axis.ticks.y = element_line(colour = "black"),
        legend.title = element_text(size = 20),
        legend.text = element_text(size = 20),
        strip.text.x = element_text(size = 18),
        strip.text.y = element_text(size = 16)) +
    guides(fill = guide_legend(nrow = 1))
ggsave('figures/correlation_plot.pdf', width=20, height=10)

## learning curve figure
human <- read_csv("data/human.csv") %>%
    filter(Block <= 16) %>%
    rename(
        type = Type,
        block = Block,
        error = Error,
        prob_correct = `Probability Correct`
    ) %>%
    select(type, block, prob_correct)

ggplot(human, aes(x = block, y = prob_correct, color = factor(type))) +
    geom_line() +
    theme_few()

## sample learning curves from successful simulations for each model type
n_samples <- 5
alcove_attn_sample_df <- alcove_all_orders %>%
    filter(correct, lr_attention >= 0) %>%
    ungroup() %>%
    sample_n(n_samples) %>%
    mutate(idx = 1:n_samples, model = 'CNN-ALCOVE-Attn') %>%
    left_join(alcove_all_df)
alcove_no_attn_sample_df <- alcove_all_orders %>%
    filter(correct, lr_attention == 0) %>%
    ungroup() %>%
    sample_n(n_samples) %>%
    mutate(idx = 1:n_samples, model = 'CNN-ALCOVE-No-Attn') %>%
    left_join(alcove_all_df)
mlp_sample_df <- mlp_all_orders %>%
    filter(correct) %>%
    ungroup() %>%
    sample_n(n_samples) %>%
    mutate(idx = 1:n_samples, model = 'CNN-MLP') %>%
    left_join(mlp_all_df)
sample_df <- bind_rows(list(alcove_attn_sample_df, alcove_no_attn_sample_df, mlp_sample_df))

ggplot(sample_df, aes(x = epoch, y = prob_correct, color = factor(type))) +
    geom_line(size=1) +
    facet_grid(idx ~ model) +
    xlab("Epoch") +
    ylab("Probability Correct") +
    scale_color_discrete("Category Type") +
    scale_x_continuous(breaks=seq(0,128,32)) +
    theme_few() +
    theme(strip.background = element_blank(),
        legend.position = "bottom",
        panel.border = element_rect(colour = "black", size = 1),
        axis.title.x = element_text(size = 20),
        axis.title.y = element_text(size = 20),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),
        axis.ticks.x = element_line(colour = "black"),
        axis.ticks.y = element_line(colour = "black"),
        legend.title = element_text(size = 20),
        legend.text = element_text(size = 20),
        strip.text.x = element_text(size = 18),
        strip.text.y = element_blank()) +
    guides(color = guide_legend(nrow = 1))
ggsave('figures/learning-curves.pdf', width=10, height=12)

## type I analyses
type_one_mlp <- alcove_all_df %>%
    filter(image_set == 'shj_images_set2', c == 5.0, phi == 5.0)

ggplot(type_one_mlp, aes(x = epoch, y = prob_correct, color = factor(type))) +
    geom_polygon()
    geom_line() +
    facet_grid(lr_association + lr_attention ~ net_type + permutation) +
    theme_few()

## shj image set 2 analyses
images_two_df <- alcove_all_orders %>%
    left_join(alcove_all_df) %>%
    filter(image_set == 'shj_images_set2') %>%
    filter(correct)

ggplot(images_two_df, aes(x = epoch, y = prob_correct, color = factor(type))) +
    geom_line() +
    theme_few() +
    facet_wrap(~ net_type + lr_association + lr_attention + c + phi + permutation) +
    theme(strip.text.x = element_blank())
