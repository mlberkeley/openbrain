package wowbrain;

import com.ericsson.otp.erlang.*;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;

public class ScreenGrabberDude{
    public static void sendScreenPixels(){
        while(true){
            try{
                Thread.sleep(500);
            } catch (InterruptedException e){

            }

            Rectangle screenRect = new Rectangle(Toolkit.getDefaultToolkit().getScreenSize());
            BufferedImage capture = null;
            try {
                capture = new Robot().createScreenCapture(screenRect);
            } catch (AWTException e) {
                e.printStackTrace();
            }

//			System.out.println("about to send to erlang");
            sendPixelsToErlang(capture);
        }
    }

    public static BufferedImage getGrayScale(BufferedImage inputImage){
        BufferedImage img = new BufferedImage(inputImage.getWidth(), inputImage.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics g = img.getGraphics();
        g.drawImage(inputImage, 0, 0, null);
        g.dispose();
        return img;
    }

    public static void sendPixelsToErlang(BufferedImage img){
        WritableRaster raster = getGrayScale(img).getRaster();


        DataBufferByte data = (DataBufferByte) raster.getDataBuffer();

        byte[] pix = data.getData();

        final int POINTS = 500;

        OtpErlangObject[] erlArr = new OtpErlangObject[POINTS];

//		TODO fix sampling
        int sample_interval = pix.length / POINTS;

        for(int i = 0; i < POINTS; i++){
            erlArr[i] = new OtpErlangByte(pix[i * sample_interval]);
        }

        OtpErlangObject[] reply = new OtpErlangObject[2];

        OtpErlangList erlList = new OtpErlangList(erlArr);

        reply[0] = new OtpErlangAtom("pixels");

        reply[1] = erlList;

        OtpErlangTuple myTuple = new OtpErlangTuple(reply);

        Robotter.myOtpMbox.send(Robotter.pixel_pid, myTuple);
    }
}
