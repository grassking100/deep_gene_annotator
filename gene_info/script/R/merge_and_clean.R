library(plyr)
library(dplyr)
get_five_end <- function(data,start,end,as_numeric=T){
  five_end=apply(data,1,function(temp){
    if(temp['strand'] == '+'){
      return(temp[start])
    }
    else{
      return(temp[end])
    }}
  )
  if(as_numeric){
    return(as.numeric(five_end))
  }
  else{
    return(five_end)
  }
}
get_three_end <- function(data,start,end,as_numeric=T){
  three_end <- apply(data,1,function(temp){
    if (temp['strand'] == '+'){
      return(temp[end])
    }
    else{
      return(temp[start])
    }}
  )
  if (as_numeric){
    return(as.numeric(three_end))
  }
  else{
    return(three_end)
  }
}
consist_ <- function(data,by,ref_value,remove_duplicate){
  return(plyr::ddply(data, by,
               function(x){
                 value <- as.numeric(unlist(x[ref_value]))
                 max_index <- which(value==max(value))
                 if(length(max_index)==1 | !remove_duplicate)
                 {
                   return(unique(x[max_index,]))
                 }
               }))
}
consist <- function(data,by,ref_value,remove_duplicate=T){
  strands <- unique(data$strand)
  chrs <- unique(data$chr)
  return_data <- NA
  for(strand_ in strands){
    for(chr_ in chrs){
      subdata <- subset(data,chr==chr_ & strand==strand_)
      consist_data = consist_(subdata,by,ref_value,remove_duplicate)
      if(!is.data.frame(return_data) )
      {
        return_data <- consist_data
      }
      else
      {
        return_data <- rbind(consist_data,return_data)
      }
    }
  }
  return(return_data)
}
coordinate_merger <- function(data,by,concat_columns){
  return(plyr::ddply(data, by,
               function(x){
                 returned=c()
                 for(column in concat_columns){
                   concated <- paste(unique(unlist(x[column])), collapse="_")
                   returned[[paste('merged_',column,sep ='')]]=concated
                 }
                 return(returned)
               }))
}


dirty_filter <- function(data,by,source_name,want_sources,dirty_sources,ref_value){
  return(plyr::ddply(data, by,
               function(x){
                 wanted_data <- x[x[,source_name] %in% want_sources,]
                 dirty_data <- x[x[,source_name] %in% dirty_sources,]
                 if(nrow(wanted_data)>1 | nrow(dirty_data)>1)
                 {
                   stop("The data has multiple sites")
                 }
                 else
                 {
                   if(nrow(wanted_data)==1)
                   {
                     if(nrow(dirty_data)==1)
                     {
                       want_value <- as.numeric(unlist(wanted_data[,ref_value]))
                       dirty_value <- as.numeric(unlist(dirty_data[,ref_value]))
                       if(want_value>dirty_value)
                       {
                         return(wanted_data)
                       }
                     }
                     else
                     {
                       return(wanted_data)
                     }
                   }
                 }
               }
  )
  )
  return(return_data)
}
reorder_merge_bed <- function(bed){
  new_bed <- apply(bed,1,function(data){
    start <- as.numeric(unlist(strsplit(as.character(data['block_related_start']),",")))
    size <- as.numeric(unlist(strsplit(as.character(data['block_size']),",")))
    start_order <- order(start)
    starts <- start[start_order]
    sizes <- size[start_order]
    count <- as.numeric(data['count'])
    previous_end <- -2
    new_starts <- c()
    new_sizes <- c()
    for(i in 1:count)
    {
      start <- starts[i]
      size <- sizes[i]
      end = start + size - 1
      if (start == (previous_end+1))
      {
        new_sizes[length(new_sizes)] <- (new_sizes[length(new_sizes)] + size)
      }
      else
      {
        new_sizes <- c(new_sizes,size)
        new_starts <- c(new_starts,start)
      }
      previous_end <- end
    }
    data['count'] <- length(new_starts)
    
    data['block_related_start'] <- str_c(new_starts,collapse=",")
    data['block_size'] <- str_c(new_sizes,collapse=",")
    return(data)
  })
  return( as.data.frame(t(new_bed)))
}
coordinate_consist_filter_ <- function(data,group_by,site){
  return(plyr::ddply(data, group_by,
               function(x){
                 value <- as.numeric(unlist(x[site]))
                 if(all(value==value[1]))
                 {
                   return(x)
                 }
               }))
}
coordinate_consist_filter <- function(data,group_by,site){
  strands <- unique(data$strand)
  chrs <- unique(data$chr)
  return_data <- NA
  for(strand_ in strands){
    for(chr_ in chrs){
      subdata <- subset(data,chr==chr_ & strand==strand_)
      consist_data = coordinate_consist_filter_(subdata,group_by,site)
      if(!is.data.frame(return_data) )
      {
        return_data <- consist_data
      }
      else
      {
        return_data <- rbind(consist_data,return_data)
      }
    }
  }
  return(return_data)
}