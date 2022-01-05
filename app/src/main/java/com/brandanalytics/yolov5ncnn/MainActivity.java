package com.brandanalytics.yolov5ncnn;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.Dialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.drawable.ColorDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.provider.MediaStore;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.view.Display;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.github.chrisbanes.photoview.PhotoViewAttacher;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.InputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends Activity
{
    //Static variables
    private final YoloV5Ncnn yolov5ncnn = new YoloV5Ncnn();
    private static final int PERMISSION_REQUEST_CODE = 1;
    private static final int SELECT_IMAGE = 1, CAMERA_REQUEST = 2;
    private static final String[] permissions = {Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.CAMERA};

    //Variables for Detect feature
    private ImageView imageView;
    private Button buttonImage, buttonCamera, buttonDetect;
    private Bitmap bitmap = null;
    private Bitmap loadImg = null;
    private MappedByteBuffer modelBuffer;
    private Thread bgThread, bgDirThread, bgAssetThread, bgYoloInitThread;
    private YoloV5Ncnn.Obj[] detections=null;
    private Handler detectHandler, yoloInitHandler;
    private String savedPath,saveImageName;
    private boolean retYoloInit;

    private PhotoViewAttacher mAttacher;

    //Variables for Experimental inference
    private static ArrayList<String> imageFiles = new ArrayList<String>();
    private static int totalPreds = 0;
    private String tlabel;
    private static int tp[] = {0,0,0,0} , fp[] = {0,0,0,0}, fn[] = {0,0,0,0};
    private final ArrayList<String> classNames= new ArrayList<String>(Arrays.asList("none","adidas","nike","asics"));


    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        if (!checkPermission()) {
            requestPermissions(permissions, PERMISSION_REQUEST_CODE);
            Log.e("MainActivity", "All permissions not acquired");
        }
        else {
            onCreateWithPermission(1);
        }
    }

    @SuppressLint({"HandlerLeak", "ClickableViewAccessibility"})
    private void onCreateWithPermission(){
        setContentView(R.layout.main);

        /*if (!OpenCVLoader.initDebug()) {
            Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
        } else {
            Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
        }*/

        runLoadAssetsThread();
        try {
            bgAssetThread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        runYoloInitThread();

        imageView = findViewById(R.id.imageView);
        buttonImage = findViewById(R.id.buttonImage);
        buttonDetect = findViewById(R.id.buttonDetect);
        buttonCamera = findViewById(R.id.buttonCamera);

        imageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Display display = getWindowManager().getDefaultDisplay();
                int width = display.getWidth();
                int height = display.getHeight();
                previewZoom(imageView,width,height);
            }
        });

        buttonImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                Intent i = new Intent(Intent.ACTION_PICK);
                i.setType("image/*");
                startActivityForResult(i, SELECT_IMAGE);
            }
        });

        buttonCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                Intent i = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(i, CAMERA_REQUEST);
            }
        });

        buttonDetect.setOnClickListener(new View.OnClickListener() {
            @SuppressLint("HandlerLeak")
            @Override
            public void onClick(View arg0) {
                if (bitmap == null)
                    return;
                buttonDetect.setEnabled(false); //disable until detect finished
                buttonImage.setEnabled(false); //disable until detect finished
                buttonCamera.setEnabled(false); //disable until detect finished
                runSaveDirThread();
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        imageView.setImageBitmap(loadImg);
                        Log.d("Update Image","Yes");
                    }
                });

                try {
                    bgYoloInitThread.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                runDetectThread();

                detectHandler = new Handler(){
                    @Override
                    public void handleMessage(Message msg) {
                        if (msg.what == 1) {
                            showObjects();
                        }else{
                            Toast emptyDet = Toast.makeText(getApplicationContext(),"Zero Detections!!",Toast.LENGTH_LONG);
                            LinearLayout layout = (LinearLayout) emptyDet.getView();
                            if (layout.getChildCount() > 0) {
                                TextView tv = (TextView) layout.getChildAt(0);
                                tv.setTextSize(30);
                            }
                            emptyDet.setGravity(Gravity.CENTER, 0, 0);
                            emptyDet.show();
                            imageView.setImageBitmap(bitmap);
                        }
                        buttonImage.setEnabled(true);
                        buttonCamera.setEnabled(true);
                    }
                };
            }
        });

        yoloInitHandler = new Handler(){
            @Override
            public void handleMessage(Message msg) {
                if (msg.what == 1 && bitmap != null) {
                    buttonDetect.setEnabled(true);
                }
            }
        };
    }

    //Experimental Batch Inference
    @SuppressLint({"HandlerLeak", "ClickableViewAccessibility"})
    private void onCreateWithPermission(int overload){
        setContentView(R.layout.main);


        listFilesRecursively("experimental");
        buttonDetect = findViewById(R.id.buttonDetect);

        runLoadAssetsThread();
        try {
            bgAssetThread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        runYoloInitThread();
        yoloInitHandler = new Handler(){
            @Override
            public void handleMessage(Message msg) {}
        };
        detectHandler = new Handler() {
            @Override
            public void handleMessage(Message msg){}
        };

        try {
            bgYoloInitThread.join();
            buttonDetect.setEnabled(true);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        buttonDetect.setOnClickListener(new View.OnClickListener() {
            @SuppressLint("HandlerLeak")
            @Override
            public void onClick(View arg0) {
                for (int i = 0; i < imageFiles.size(); i++) {
                    Log.d("Experimental Inference", String.format("Image {%d/%d}: %s\n", i, imageFiles.size(), imageFiles.get(i)));
                    InputStream loadbitmap = null;
                    Log.d("Experimental Inference/Filename", imageFiles.get(i));
                    try {
                        tlabel = imageFiles.get(i).split("/")[1];
                        loadbitmap = getAssets().open(imageFiles.get(i));
                        bitmap = BitmapFactory.decodeStream(loadbitmap);
                        loadbitmap.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    if (bitmap == null)
                        return;
                    buttonDetect.setEnabled(false); //disable until detect finished

                    try {
                        bgYoloInitThread.join();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }

                    runDetectThread();
                    try {
                        bgThread.join();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    if (detections.length > 0) {
                        totalPreds += detections.length;
                        for (YoloV5Ncnn.Obj d : detections) {
                            if (d.label.equals(tlabel))
                                tp[classNames.indexOf(d.label)] += 1;
                            else {
                                fp[classNames.indexOf(d.label)] += 1;
                                fn[classNames.indexOf(tlabel)] += 1;
                            }
                        }
                    }
                    else {
                        Log.d("Experimental Inference", "No Detections");
                    }
                    bitmap = null;
                }
                Log.i("Experimetal Inference/TP",Arrays.toString(tp));
                Log.i("Experimetal Inference/FP",Arrays.toString(fp));
                Log.i("Experimetal Inference/FN",Arrays.toString(fn));
                float totTP=0;
                for(float t:tp) totTP+=t;
                Log.i("Experimetal Inference/Acc:",String.valueOf(totTP/totalPreds));
            }
        });
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if(requestCode == PERMISSION_REQUEST_CODE && grantResults.length == 3){
            onCreateWithPermission();
        }
    }

    public boolean checkPermission() {
        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            if(ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED)
                return ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
        return false;
    }

    private void showObjects()
    {
        if (detections == null)
        {
            imageView.setImageBitmap(bitmap);
            return;
        }

        // draw objects on bitmap
        Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        int h = rgba.getHeight();
        int w = rgba.getWidth();
        final int[] colors = new int[] {
            Color.rgb( 54,  67, 244),
            Color.rgb( 99,  30, 233),
            Color.rgb(176,  39, 156),
            Color.rgb(183,  58, 103),
            Color.rgb(181,  81,  63),
            Color.rgb(243, 150,  33),
            Color.rgb(244, 169,   3),
            Color.rgb(212, 188,   0),
            Color.rgb(136, 150,   0),
            Color.rgb( 80, 175,  76),
            Color.rgb( 74, 195, 139),
            Color.rgb( 57, 220, 205),
            Color.rgb( 59, 235, 255),
            Color.rgb(  7, 193, 255),
            Color.rgb(  0, 152, 255),
            Color.rgb( 34,  87, 255),
            Color.rgb( 72,  85, 121),
            Color.rgb(158, 158, 158),
            Color.rgb(139, 125,  96)
        };

        Canvas canvas = new Canvas(rgba);

        Paint paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(4);

        Paint textbgpaint = new Paint();
        textbgpaint.setColor(Color.WHITE);
        textbgpaint.setStyle(Paint.Style.FILL);

        Paint textpaint = new Paint();
        textpaint.setColor(Color.BLACK);
        textpaint.setTextSize(26);
        textpaint.setTextAlign(Paint.Align.LEFT);

        for (int i = 0; i < detections.length; i++)
        {
            paint.setColor(colors[i % 19]);

            float x1 = (detections[i].x - detections[i].w/2)*w;
            float x2 = (detections[i].x + detections[i].w/2)*w;
            float y1 = (detections[i].y - detections[i].h/2)*h;
            float y2 = (detections[i].y + detections[i].h/2)*h;

            canvas.drawRect(x1, y1, x2, y2, paint);

            // draw filled text inside image
            {
                String text = detections[i].label + " = " + String.format(Locale.US,"%.1f", detections[i].prob * 100) + "%";

                float text_width = textpaint.measureText(text);
                float text_height = - textpaint.ascent() + textpaint.descent();

                float x = x1;
                float y = y1 - text_height;
                if (y < 0)
                    y = 0;
                if (x + text_width > rgba.getWidth())
                    x = rgba.getWidth() - text_width;

                canvas.drawRect(x, y, x + text_width, y + text_height, textbgpaint);

                canvas.drawText(text, x, y - textpaint.ascent(), textpaint);
            }
        }

        imageView.setImageBitmap(rgba);
        if(String.valueOf(bgDirThread.getState()).equals("TERMINATED")) {
            try (FileOutputStream out = new FileOutputStream(savedPath)) {
                rgba.compress(Bitmap.CompressFormat.PNG, 100, out);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == SELECT_IMAGE && resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();
            try {
                if (requestCode == SELECT_IMAGE) {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),selectedImage);
                    imageView.setImageBitmap(bitmap);
                    if (selectedImage.getScheme().equals("file")) {
                        saveImageName = selectedImage.getLastPathSegment();
                    } else {
                        Cursor cursor = null;
                        try {
                            cursor = getContentResolver().query(selectedImage, new String[]{
                                    MediaStore.Images.ImageColumns.DISPLAY_NAME
                            }, null, null, null);

                            if (cursor != null && cursor.moveToFirst()) {
                                saveImageName = cursor.getString(cursor.getColumnIndex(MediaStore.Images.ImageColumns.DISPLAY_NAME));
                                Log.d("Input Filename", "name is " + saveImageName);
                            }
                        } finally {

                            if (cursor != null) {
                                cursor.close();
                            }
                        }
                    }
                }
            }
            catch (IOException e) {
                Log.e("MainActivity", "FileNotFoundException");
            }
        }
        if (requestCode == CAMERA_REQUEST && resultCode == RESULT_OK && null != data) {
            bitmap = (Bitmap) data.getExtras().get("data");
            imageView.setImageBitmap(bitmap);
        }
        if(retYoloInit && bitmap != null){
            buttonDetect.setEnabled(true);
        }
    }

    private void runDetectThread(){
        bgThread = new Thread(new Runnable() {
            @Override
            public void run()
            {
                detections = yolov5ncnn.Detect(bitmap);
                Log.d("Detect Count", String.valueOf(detections.length));
                if(detections != null)
                    if(detections.length > 0)
                        detectHandler.sendEmptyMessage(1);
                    else
                        detectHandler.sendEmptyMessage(0);
            }
        });
        bgThread.setName("detectThread");
        bgThread.start();
        Log.d("Detect Thread State", String.valueOf(bgThread.getState()));
    }

    private void runSaveDirThread(){
        bgDirThread = new Thread(new Runnable() {
            @Override
            public void run()
            {
                File baseDir = new File("/storage/emulated/0/Pictures/BrandAnalytics");
                if(saveImageName == null) {
                    String mTimeStamp = new SimpleDateFormat("ddMMyyyy_HHmmssSS", Locale.US).format(new Date());
                    saveImageName = "CAP_"+mTimeStamp+".jpg";
                }
                File saveFilePath = new File(baseDir, "DET_"+saveImageName);
                try{
                    if (!baseDir.exists()) {
                        boolean wasSuccessful = baseDir.mkdir();
                        if (!wasSuccessful) {
                            System.out.println("Save Dir not created");
                        }
                    }
                    if (!saveFilePath.exists()) {
                        boolean wasSuccessful = saveFilePath.createNewFile();
                        if (!wasSuccessful) {
                            System.out.println("File not created");
                        }
                    }
                    savedPath = saveFilePath.toString();
                }
                catch (IOException e) {
                    e.printStackTrace();
                }
                finally {
                    saveImageName = null;
                }
            }
        });
        bgDirThread.setName("SaveDirThread");
        bgDirThread.start();
        try {
            bgDirThread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void runLoadAssetsThread(){
        final String modelPath = "best-uint8.tflite";
        bgAssetThread = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    AssetFileDescriptor fileDescriptor = getResources().getAssets().openFd(modelPath);
                    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
                    FileChannel fileChannel = inputStream.getChannel();
                    long startOffset = fileDescriptor.getStartOffset();
                    long declaredLength = fileDescriptor.getDeclaredLength();
                    modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

                    InputStream loadbitmap = getAssets().open("loading.jpg");
                    loadImg = BitmapFactory.decodeStream(loadbitmap);
                    loadbitmap.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });
        bgAssetThread.setName("LoadAssetThread");
        bgAssetThread.start();
    }

    private void runYoloInitThread(){
        bgYoloInitThread = new Thread(new Runnable() {
            @Override
            public void run() {
                retYoloInit = yolov5ncnn.Init(getAssets(), modelBuffer);
                if (!retYoloInit) {
                    Log.e("MainActivity", "BA Init failed");
                }
                else{
                    yoloInitHandler.sendEmptyMessage(1);
                }
            }
        });
        bgYoloInitThread.setName("YoloInitThread");
        bgYoloInitThread.start();
    }

    private void previewZoom(ImageView imageView, int width, int height) {

        final Dialog dialog = new Dialog(this);
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        dialog.getWindow().setBackgroundDrawable(new ColorDrawable(android.graphics.Color.TRANSPARENT));
        LayoutInflater inflater = (LayoutInflater) this.getSystemService(LAYOUT_INFLATER_SERVICE);
        View layout = inflater.inflate(R.layout.custom_fullimage_dialog,
                (ViewGroup) findViewById(R.id.layout_root));
        ImageView image = (ImageView) layout.findViewById(R.id.fullimage);
        image.setImageDrawable(imageView.getDrawable());
        image.getLayoutParams().height = height;
        image.getLayoutParams().width = width;
        mAttacher = new PhotoViewAttacher(image);
        image.requestLayout();
        dialog.setContentView(layout);
        dialog.show();
    }

    private boolean listFilesRecursively(String path){
        String [] list;
        try {
            list = getAssets().list(path);
            if (list.length > 0) {
                for (String file : list) {
                    if (!listFilesRecursively(path + "/" + file))
                        return false;
                    else {
                        if(file.endsWith(".jpg") || file.endsWith(".png") || file.endsWith(".jpeg"))
                            imageFiles.add(path + "/" + file);
                    }
                }
            }
        } catch (IOException e) {
            return false;
        }
        return true;
    }
}
