simple_belong_by_boundary <- function(exp_sites,boundarys,start_name,end_name){
  lbs <- apply(boundarys[,c(start_name,end_name)],1,min)
  ubs <- apply(boundarys[,c(start_name,end_name)],1,max)
  boundary_count <- nrow(boundarys)
  exp_count <- length(exp_sites)
  belong_indice <- rep(NA,exp_count)
  for(boundary_index in 1:boundary_count) {
    lb <- lbs[boundary_index]
    ub <- ubs[boundary_index]
    for(exp_index in 1:exp_count){
      exp_site <- as.numeric(exp_sites[exp_index])
      if (lb <= exp_site & ub >= exp_site){
        if(is.na(belong_indice[exp_index]))
        {
          belong_indice[exp_index]=as.character(boundary_index)
        }
        else{
          belong_indice[exp_index] <- paste(belong_indice[exp_index],as.character(boundary_index),sep=',')
        }
      }
    }
  }
  return(belong_indice)
}

belong_by_boundary <- function(exp_sites,boundarys,exp_site_name,boundary_start_name,
                               boundary_end_name,exp_name,ref_name){
  strands <- unique(exp_sites$strand)
  chrs <- unique(exp_sites$chr)
  returned_data <- data.frame(matrix(ncol = ncol(exp_sites)+1, nrow = 0),stringsAsFactors = F)
  for(strand_ in strands)
  {
    for(chr_ in chrs)
    {
      print(paste0(chr_," ",strand_))
      selected_exp_sites <- subset(exp_sites,strand == strand_ & chr == chr_)
      selected_boundarys <- subset(boundarys,strand == strand_ & chr == chr_)
      if(nrow(selected_exp_sites)>0)
      {
        if(nrow(selected_boundarys)>0)
        {
          belong_indice <- simple_belong_by_boundary(selected_exp_sites[,exp_site_name],selected_boundarys,
                                                     boundary_start_name,boundary_end_name)
          for(i in 1:length(belong_indice))
          {
            index = belong_indice[i]
            selected_exp_site = selected_exp_sites[i,]
            for(sub_index in strsplit(index,split=',')[[1]])
            {
              name=selected_boundarys[as.numeric(sub_index),ref_name]
                #print(name)
              selected_exp_site_ <- cbind(selected_exp_site,ref_name=name)
              returned_data <- rbind(returned_data,selected_exp_site_)
            }
          }
        }
        else
        {
          selected_exp_site_ <- cbind(selected_exp_sites,ref_name=NA)
          returned_data <- rbind(returned_data,selected_exp_site_)
        }
      }
    } 
  }
  return(returned_data)
}

simple_belong_by_distance <- function(exp_sites,ref_sites,
                                      upstream_dist,downstream_dist){
  max_ref_index <- length(ref_sites)
  max_exp_index <- length(exp_sites)
  belong_indice <- rep(NA,max_exp_index)
  for (ref_index in 1:max_ref_index){
    ref_site <- as.numeric(ref_sites[ref_index])
    for (exp_index in 1:max_exp_index){
      exp_site <- as.numeric(exp_sites[exp_index])
      ub = ref_site + upstream_dist
      db = ref_site + downstream_dist
      if (ub<=exp_site & exp_site <=db)
      {
        if(is.na(belong_indice[exp_index]))
        {
          belong_indice[exp_index]=as.character(ref_index)
        }
        else
        {
          belong_indice[exp_index] <- paste(belong_indice[exp_index],as.character(ref_index),sep=',')
        }
      }
    }
  }
  return(belong_indice)
}

belong_by_distance <- function(exp_sites,ref_sites,five_dist,three_dist,
                               exp_site_name,ref_site_name,exp_name,ref_name){
  returned_data <- data.frame(matrix(ncol = ncol(exp_sites)+1, nrow = 0),stringsAsFactors = F)
  strands <- unique(exp_sites$strand)
  chrs <- unique(exp_sites$chr)
  for(strand_ in strands){
    for(chr_ in chrs){
      print(paste0(chr_," ",strand_))
      selected_exp_sites <- subset(exp_sites,strand == strand_ & chr == chr_)
      selected_ref_sites <- subset(ref_sites,strand == strand_ & chr == chr_)
      if(nrow(selected_exp_sites)>0)
      {
        if(nrow(selected_ref_sites)>0)
        {
          if (strand_ == '+'){
            upstream_dist <- five_dist
            downstream_dist <- three_dist
          }
          else
          {
            upstream_dist <- -three_dist
            downstream_dist <- -five_dist
          }
          belong_indice <- simple_belong_by_distance(selected_exp_sites[,exp_site_name],
                                                    selected_ref_sites[,ref_site_name],
                                                    upstream_dist,downstream_dist)
          for(i in 1:length(belong_indice))
          {
            index = belong_indice[i]
            selected_exp_site = selected_exp_sites[i,]
            if(!is.na(index)){
              for(sub_index in strsplit(index,split=',')[[1]])
              {
                if(!is.na(sub_index))
                {
                    selected_exp_site_ <- cbind(selected_exp_site,ref_name=selected_ref_sites[as.numeric(sub_index),ref_name])
                    returned_data <- rbind(returned_data,selected_exp_site_)
                }
              }
            }
          }        
        }
        else
        {
          selected_exp_site_ <- cbind(selected_exp_sites,ref_name=NA)
          returned_data <- rbind(returned_data,selected_exp_site_)
        }  
      }
    } 
  }
  return(returned_data)
}