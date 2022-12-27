import base64
import io

def plt2base64(plt):
    pic_io = io.BytesIO()
    plt.savefig(pic_io,  format='png')
    pic_io.seek(0)
    pic_hash = base64.b64encode(pic_io.read())
    return bytes.decode(pic_hash) # convert bytes to string

def plt2html(plt):
    '''
    output an HTML img tag
    '''
    return '<img src="data:image/.png;base64,' + plt2base64(plt) + '">'
