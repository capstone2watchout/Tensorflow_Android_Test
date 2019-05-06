package com.example.tensorflow_test;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {


    private static final String INPUT_NAME = "Placeholder";
    private static final String OUTPUT_NAME = "final_result";
    private static final String [] OUTPUT_NODES = {"daisy", "dandelion", "roses", "sunflowers", "tulips"};
    private static final String MODEL_FILE = "file:///android_asset/stripped.pb";
    private static final String LABEL_FILE = "file:///android_asset/out_label.txt";

    private static int INPUT_SIZE = 299;
    private static int IMAGE_MEAN = 0;
    private static float IMAGE_STD = 255.0f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        TensorFlowInferenceInterface inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);

        int [] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        float [] floatValues = new float[INPUT_SIZE * INPUT_SIZE * 3];

        final Operation operation = inferenceInterface.graphOperation(OUTPUT_NAME);
        final int numClasses = (int) operation.output(0).shape().size(1);

        float [] outputs = new float[numClasses];

        Bitmap bitmap;

        bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.tulips);
        bitmap = bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
        System.out.println(bitmap.getWidth() + " " + bitmap.getHeight() + " " + numClasses);

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
            floatValues[i * 3 + 2] = ((val & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
        }

        inferenceInterface.feed(INPUT_NAME, floatValues, 1, INPUT_SIZE, INPUT_SIZE, 3);
        inferenceInterface.run(new String[] {OUTPUT_NAME}, false);
        inferenceInterface.fetch(OUTPUT_NAME, outputs);

        System.out.println("===============");
        for (int i=0; i<numClasses; i++)
            System.out.println(OUTPUT_NODES[i] + " "  + outputs[i]);

    }
}
