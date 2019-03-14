library('gsubfn')
library('plyr')
get_parent <- function(gff_group_column){
  result <- strapplyc(gff_group_column, '(Parent=.+?;)')
  return(as.character(lapply(result,function(d)substr(d,8,nchar(d)-1))))
}

get_id <- function(gff_group_column){
  result <- strapplyc(gff_group_column, '(ID=.+?;)')
  return(as.character(lapply(result,function(d)substr(d,4,nchar(d)-1))))
}

id_in_list <- function(ids,id_list){
  result <- unlist(strsplit(ids, ','))
  return(any(result %in% id_list))
}

write_bed12 <- function(bed,path,...){
  bed_ <- bed
  bed_['start'] <- as.numeric(as.character(unlist(bed_['start']))) - 1
  bed_['orf_start'] <- as.numeric(as.character(unlist(bed_['orf_start']))) - 1
  bed_ <-format(bed_, scientific = F,trim=T)
  write.table(bed_,path,sep='\t', quote = F,col.names =F, row.names =F,...)
}
write_bed8 <- function(bed,path,...){
  bed_ <- bed
  bed_['start'] <- as.numeric(as.character(unlist(bed_['start']))) - 1
  bed_['orf_start'] <- as.numeric(as.character(unlist(bed_['orf_start']))) - 1
  bed_ <-format(bed_, scientific = F,trim=T)
  write.table(bed_,path,sep='\t', quote = F,col.names =F, row.names =F,...)
}
write_bed <- function(bed,path,...){
  bed_ <- bed
  bed_['start'] <- as.numeric(as.character(unlist(bed_['start']))) - 1
  bed_ <-format(bed_, scientific = F,trim=T) 
  write.table(bed_,path,sep='\t', quote = F,col.names =F, row.names =F,...)
  
}


write_gff <- function(gff,path,...){
  gff <-format(gff, scientific = FALSE,trim=T) 
  write.table(gff,path,sep='\t', quote = F,col.names =F, row.names =F,...)
}

find_five_most_UTR <- function(gff){
  start <- as.numeric(unlist(gff['start']))
  end <- as.numeric(unlist(gff['end']))
  if(gff[1,'strand']=='+'){
    return(list(start=min(start),end=min(end)))
  }
  else{
    return(list(start=max(end),end=max(start)))
  }
}

find_three_most_UTR <- function(gff){
  start <- as.numeric(unlist(gff['start']))
  end <- as.numeric(unlist(gff['end']))
  if(gff[1,'strand']=='+'){
    return(list(start=max(start),end=max(end)))
  }
  else{
    return(list(start=min(end),end=min(start)))
  }
}

read_gff <- function(path)
{
  gff <- read.csv(path,sep='\t',header=F,stringsAsFactors = F,comment.char = '#')
  parents <- get_parent(gff$V9)
  ids <- get_id(gff$V9)
  gff <- cbind.data.frame(gff,parent=parents,stringsAsFactors = F)
  gff <- cbind.data.frame(gff,id=ids,stringsAsFactors = F)
  gff$V9='.'
  colnames(gff) <- c('chr','source','feature','start','end','score','strand','frame','attribute','parent','id')
  return(gff)
}

read_bed8 <- function(path)
{
  bed <- read.csv(path,sep='\t',header=F,stringsAsFactors = F,comment.char = '#')
  colnames(bed) <- c('chr','start','end','id','score','strand','orf_start','orf_end')
  bed['start'] <- bed['start'] + 1
  bed['orf_start'] <- bed['orf_start'] + 1
  return(bed)
}

read_bed12 <- function(path)
{
  bed <- read.csv(path,sep='\t',header=F,stringsAsFactors = F,comment.char = '#')
  colnames(bed) <- c('chr','start','end','id','score','strand','orf_start','orf_end',
                     'rgb','count','block_size','block_related_start')
  bed['start'] <- bed['start'] + 1
  bed['orf_start'] <- bed['orf_start'] + 1
  return(bed)
}

read_bed <- function(path)
{
  bed <- read.csv(path,sep='\t',header=F,stringsAsFactors = F,comment.char = '#')
  colnames(bed) <- c('chr','start','end','id','score','strand')
  bed['start'] <- bed['start'] + 1
  return(bed)
}


gff_to_bed <- function(gff,id_name='id',orf_starts=NA,orf_ends=NA){
  bed <- gff[,c('chr','start','end',id_name,'score','strand')]
  if(typeof(orf_starts)=='double' &  typeof(orf_ends)=='double')
  {
    bed <- cbind(bed,orf_start=find_orf_start(bed[,id_name]),
                 orf_end=find_orf_end(bed[,id_name]))
  }
  
  return(unique(bed))
}

