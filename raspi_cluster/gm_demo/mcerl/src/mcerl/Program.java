package mcerl;

import java.awt.AWTException;
import java.awt.FlowLayout;
import java.awt.Frame;
import java.awt.Rectangle;
import java.awt.Robot;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
 
/**
 * This program demonstrates how to capture a screenshot (full screen)
 * as an image which will be saved into a file.
 * @author www.codejava.net
 *
 */
public class Program {
 
    public static BufferedImage capture(){
        try{    
            Robot robot = new Robot();
            String format = "jpg";
            String fileName = "FullScreenshot." + format;
             
            Rectangle screenRect = new Rectangle(50,241, 856, 482);
            BufferedImage screenFullImage = robot.createScreenCapture(screenRect);
            return screenFullImage;
        } catch (AWTException  ex) {
            System.err.println(ex);
            return null;
        }

    }
    
    public static void main(String[] args) {
        JLabel img = new JLabel(new ImageIcon(capture()));
        JFrame frame = new JFrame();
        frame.getContentPane().setLayout(new FlowLayout());
        frame.getContentPane().add(img);
        frame.pack();
        frame.setVisible(true);
        
        while(true){

            
        }
    }
}