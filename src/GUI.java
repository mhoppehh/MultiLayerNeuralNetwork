import java.util.*;
import java.io.*;
import javax.swing.*;
import java.awt.*;
import java.io.BufferedReader;
import java.awt.image.BufferedImage;
import java.io.FileReader;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
public class GUI
{
    JFrame frame;
    public GUI(){
        frame = new JFrame();
    }

    public static BufferedImage createResizedCopy(Image originalImage, 
    int scaledWidth, int scaledHeight, 
    boolean preserveAlpha)
    {
        int imageType = preserveAlpha ? BufferedImage.TYPE_INT_RGB : BufferedImage.TYPE_INT_ARGB;
        BufferedImage scaledBI = new BufferedImage(scaledWidth, scaledHeight, imageType);
        Graphics2D g = scaledBI.createGraphics();
        if (preserveAlpha) {
            g.setComposite(AlphaComposite.Src);
        }
        g.drawImage(originalImage, 0, 0, scaledWidth, scaledHeight, null); 
        g.dispose();
        return scaledBI;
    }

    public static BufferedImage getImage(Picture[] pics){
        int n = (int)(Math.ceil(Math.sqrt(pics.length)));
        BufferedImage image = new BufferedImage(n * 28 * 4,n * 28 * 4, BufferedImage.TYPE_INT_ARGB);
        Graphics g = image.getGraphics();
        for(int i = 0;i < pics.length; i++){
            BufferedImage icon = getLetterIcon(pics[i].pixels);
            g.drawImage(icon, (i % n) * 28 * 4, (i / n) * 28 * 4, null);
        }
        return image;
    }
    
    public static BufferedImage getLetterIcon(double pixels[]){
        BufferedImage image = new BufferedImage(28,28, BufferedImage.TYPE_INT_ARGB);
        Graphics g = image.getGraphics();

        for(int i = 0; i < pixels.length; i++){
            int value = (int)(255 - 255 * pixels[i]);
            image.setRGB(i % 28, i / 28, new Color(value, value, value).getRGB());
        }
        System.out.println();

        return createResizedCopy(image,(int)(28 * 4), (int)(28 * 4),false);
    }

    public void displayData(Data data, ArrayList<double[]> output){
        BufferedImage image = getImage(data.data);
        JLabel lblimage = new JLabel(new ImageIcon(image));
        lblimage.addMouseMotionListener(new MouseAdapter() {
                @Override
                public void mouseMoved(MouseEvent e) {
                    int x = e.getX();
                    int y = e.getY();
                    int index = ((x / 28)) + (y / 28) * 10;
                    String s = "";
                    if(output != null && index < output.size()){
                        for(int i = 0; i < output.get(index).length;i++){
                            s += "\n" + i + " : ";
                            for(int j = 0; j++ < output.get(index)[i] * 100;s += "*"){}
                        }
                    }
                    System.out.println('\u000C');
                    System.out.println(s);
                }
            });
        frame.getContentPane().add(lblimage, BorderLayout.CENTER);
        frame.setVisible(true);
        frame.pack();
        frame.repaint();
    }
    
    public void displayPicture(double[] pixels, double scale){
        BufferedImage image = getImage(new Picture[]{new Picture(pixels, null)});
        image = createResizedCopy(image,(int)(28 * scale), (int)(28 * scale),false);
        JLabel lblimage = new JLabel(new ImageIcon(image));
        frame.getContentPane().add(lblimage, BorderLayout.CENTER);
        frame.setVisible(true);
        frame.pack();
        frame.repaint();
    }
}
