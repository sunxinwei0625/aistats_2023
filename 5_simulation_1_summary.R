library(ggplot2)
library(reshape2)
library(dplyr)
library(latex2exp)
dir_read='/Users/5_Simulation'
setwd(dir_read)
results = read.csv('results_new.csv')
{
  datap = results
  datap_long = melt(datap, id=c('sample_size','seed'))
  datap_long$variable = factor(datap_long$variable, levels=unique(datap_long$variable), 
                               labels=c('Adjusted Effect of M1 \n', 
                                        'Adjusted Effect of M2 \n', 
                                        'Adjusted Effect of M1,M2 \n', 
                                        'Unadjusted Effect of M1 \n',
                                        'Unadjusted Effect of M2 \n', 
                                        'Unadjusted Effect of M1,M2 \n')
                               # labels=c(TeX('$\\delta_{M_1}(0)$'),TeX('$\\delta_{M_1}(1)$'),
                               #          TeX('$\\delta_{M_2}(0)$'),TeX('$\\delta_{M_2}(1)$'),
                               #          TeX('$\\delta_{M_1, M_2}(0)$'), TeX('$\\delta_{M_1, M_2}(1)$'),
                               #          TeX('$\\zeta(0)$'), TeX('$\\zeta(0)$')
                               #          )
                               )
  g=ggplot(data = datap_long[!is.na(datap_long$sample_size),])+
    geom_line(aes(x=sample_size, y=value, color=variable, group=seed), color='gray', alpha=0.2)+
    geom_point(aes(x=sample_size, y=value, color=variable, group=seed), alpha=0.1)+
    geom_hline(data = datap_long[is.na(datap_long$sample_size),], aes(yintercept=value, color=variable))+
    facet_wrap(.~variable, ncol=3, scale='free')+
    labs(x='Sample Size')+
    scale_x_continuous(breaks=c(100,500,1000,2000,5000))+
    theme(legend.position = 'none', strip.text = element_text(size=10), 
          axis.title = element_text(size=10), axis.text.x=element_text(angle=45))
  ggsave('/Users/5_Simulation/plot_simulation.pdf', g, width = 10,height=5)
}
