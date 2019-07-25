import matplotlib.pyplot as plt
import numpy as np
class SubplotHelper():
    def __init__(self,nrow=1,ncol=1,width=50,height=50):
        self.index = 0
        self._nrow=nrow
        self._ncol=ncol
        self.fig, self.axes = plt.subplots(self._nrow,self._ncol)
        self.fig.set_size_inches(width,height)
    def get_ax(self,row_index=None,col_index=None):
        if not isinstance(self.axes,np.ndarray):
            ax=self.axes
        else:
            if row_index is None or col_index is None:
                row_index = int(self.index/self._ncol)
                col_index = self.index%self._ncol
                self.index += 1
            if len(self.axes.shape)==1:
                ax=self.axes[col_index]
            else:
                ax=self.axes[row_index,col_index]
        return ax
    def set_axes_setting(self,title_params=None,xlabel_params=None,ylabel_params=None,
                         tick_params=None,legends_params=None):
        for sub_axes in self.axes:
            for ax in sub_axes:
                if title_params is not None:
                    ax.set_title(**title_params)
                if tick_params is not None:
                    ax.tick_params(**tick_params)
                if xlabel_params is not None:
                    ax.set_xlabel(**xlabel_params)
                if ylabel_params is not None:
                    ax.set_ylabel(**ylabel_params)
                if legends_params is not None:
                    ax.legend(**legends_params)
