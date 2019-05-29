#library("VennDiagram")
library(Biostrings)
hist_5_diff <- function(data,use_abs=F,...)
{
  exist <- as.numeric(data$exist_5_end)
  evidence <- as.numeric(data$evidence_5_end)
  if(use_abs)
  {
    hist(abs(evidence-exist),...)
  }else{
    hist(evidence-exist,...)
  }
  
}

hist_3_diff <- function(data,use_abs=F,...)
{
  exist <- as.numeric(data$exist_3_end)
  evidence <- as.numeric(data$evidence_3_end)
  if(use_abs)
  {
    hist(abs(evidence-exist),...)
  }else{
    hist(evidence-exist,...)
  }
}

view_nucleotide <- function(path,main_=NA){
  fasta <- readDNAStringSet(path,"fasta")
  afmc <- consensusMatrix(fasta, baseOnly=T,as.prob = T)
  matplot(t(afmc),type='l',col=c('green','blue','orange','red','black'),
          xlab='position',ylab='percentage',ylim=c(0,1),main=main_)
  legend("topright",c('A','C','G','T','N'),
         cex=0.8,fill=c('green','blue','orange','red','black'))
}
ALL_duplicate_index <- function(df){(duplicated(df) | duplicated(df, fromLast = TRUE))}

str_with_time <- function(prefix,postfix)
{
  return(paste0(prefix,format(Sys.Date(), format="%Y_%m_%d"),postfix))
}

unique_length <-function(x){length(unique(x))}