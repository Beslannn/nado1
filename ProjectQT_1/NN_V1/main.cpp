#include "mainwindow.h"
#include <QApplication>
#include <iostream>
#include <vector>
#include <QDir>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <QDebug>
#include <QTextCodec>
#include <QtSql/QSqlDatabase>
#include <QtSql>

using namespace cv;
using namespace std;

vector<double> wish_exit(94);
vector<vector<vector<double>>> matr_w_layers(3);
vector<vector<vector<double>>> error_matr_w_layers(3);
vector<vector<double>> sum_error_stol_matr_w_layers(3);
vector<vector<double>> res_layers(3);
vector<Mat> Random_learning_image(94);
vector<vector<double>> Wish_Exit(94);
double Learning_Rate = 0.1;
int kol_epoch = 25;

double Activation_Function_Sigmoid(double parameter){
    return (1/(1+exp(-parameter)));
}

double Derivative_Activation_Function_Sigmoid(double parameter1) {
    return(Activation_Function_Sigmoid(parameter1)*(1-Activation_Function_Sigmoid(parameter1)));
}

vector<double> matr_in_vect(Mat Image){
    vector<double> Res;
    Res.resize(Image.rows*Image.cols);
    for(int i = 0; i < (int)Image.rows; i++){
        for(int j = 0; j < (int)Image.cols; j++){
            Res[Image.rows*i+j] = (Image.at<Vec3b>(i, j)[0])/255;
        }
    }
    return Res;
}

vector<double> multiplication(vector<vector<double>> M_W, vector<double> V){
    vector<double> Res;
    Res.resize(M_W.size());
    vector<double> str;
    for(int i = 0; i < (int)M_W.size(); i++){
        str = M_W[i];
        for(int j = 0; j < (int)str.size(); j++){
            Res[i] += str[j]*V[i];
        }
        str.clear();
        Res[i] = Activation_Function_Sigmoid(Res[i]);
    }
    return Res;
}

vector<vector<double>> random_matr_w(int w, int h){
    vector<vector<double>> Res;
    Res.resize(h);
    srand(time(NULL));
    for (int i = 0; i < (int)Res.size(); i++){
        Res[i].resize(w);
        for (int j = 0; j < (int)Res[i].size(); j++)
            Res[i][j] = -0.5+(double)rand()/RAND_MAX;
    }
    return Res;
}

void Error_Weights(vector<vector<double>> M_W,int layer){
    sum_error_stol_matr_w_layers[layer].resize(M_W.size());
    error_matr_w_layers[layer].resize(M_W.size());
    for(int k = 0; k < (int)error_matr_w_layers[layer].size(); k++) error_matr_w_layers[layer][k].resize(M_W[k].size());
    for(int j = 0; j < (int)M_W[0].size(); j++)
        for(int i = 0; i < (int)M_W.size(); i++)
            error_matr_w_layers[layer][i][j] = sum_error_stol_matr_w_layers[layer][i]*M_W[i][j];
}

vector<double> Forward(vector<double> V_I, int parameter){
    if(parameter == 0){
        matr_w_layers[0] = random_matr_w(V_I.size(),(int)V_I.size()/2);
        matr_w_layers[1] = random_matr_w(matr_w_layers[0].size(),(int)matr_w_layers[0].size()/2);
        matr_w_layers[2] = random_matr_w(matr_w_layers[1].size(),94);
    }
    res_layers[0] = multiplication(matr_w_layers[0],V_I);
    res_layers[1] = multiplication(matr_w_layers[1],res_layers[0]);
    res_layers[2] = multiplication(matr_w_layers[2],res_layers[1]);
    return res_layers[2];
}

vector<vector<double>> Update_Weights(vector<vector<double>> Error_Weights,vector<vector<double>> Matrix_Weights, int layer){
    vector<vector<double>> Res;
    Res.resize(Matrix_Weights.size());
    for(int i = 0; i < (int)Matrix_Weights.size(); i++){
        Res[i].resize(Matrix_Weights[i].size());
        for(int j = 0; j < (int)Matrix_Weights[i].size(); j++){
            Res[i][j] = Matrix_Weights[i][j] - Learning_Rate*Error_Weights[i][j]*Derivative_Activation_Function_Sigmoid(res_layers[layer][i])*res_layers[layer-1][j];
        }
    }
    return Res;
}

void Connection_Database_MYSQL(vector<vector<double>> M_W, int layer){
    qDebug() << QSqlDatabase::drivers();

    QSqlDatabase db = QSqlDatabase::addDatabase("QMYSQL");
    db.setDatabaseName("neural_network");
    db.setUserName("root");
    db.setHostName("localhost");
    db.setPassword("170898");

    if(!db.open()) {
        qDebug() << "no: " << db.lastError();
    }
    else qDebug() << "yes";

    QSqlQuery query;
    query.exec("SELECT table_name FROM information_schema.tables");
    bool ok = false;
    while(query.next())
        if(query.value(0).toString() == "Weights_NN") ok = true;
    if(!ok)
        query.exec("CREATE TABLE Weights_NN(Number_Layers INT(1), Numbers_Str INT(4), Numbers_Stol INT(4), Value_Weight DOUBLE(10,7))");
    for(int i = 0; i < (int)M_W.size(); i++){
        for(int j = 0; j < (int)M_W[i].size(); j++){
            QString str = "INSERT INTO Weights_NN values(" + QString::number(layer) + "," + QString::number(i)  + ","  + QString::number(j)  + ",";
            str = str + QString::number(M_W[i][j]) + ")";
            query.exec(str);
        }
    }

    db.close();
}

void Read_Weights_From_Database(){
    qDebug() << QSqlDatabase::drivers();

    QSqlDatabase db = QSqlDatabase::addDatabase("QMYSQL");
    db.setDatabaseName("neural_network");
    db.setUserName("root");
    db.setHostName("localhost");
    db.setPassword("170898");
    if(!db.open()) {
        qDebug() << "no: " << db.lastError();
    }
    else qDebug() << "yes";
    QSqlQuery query;
    query.exec("SELECT * FROM Weights_NN");
    while(query.next())
        matr_w_layers[query.value(0).toInt()][query.value(1).toInt()][query.value(2).toInt()] = query.value(3).toDouble();
    db.close();
}

void Generated_learning_sequence_image(){
    QDir dir("/home/beslan/fonts/fonts png/New");
    dir.setFilter(QDir::Dirs);
    dir.setSorting(QDir::Name);
    QFileInfoList list = dir.entryInfoList();
    vector<bool> im_used;
    vector<double> wish_exit;
    for(int i = 0; i < (int)list.size(); i++) im_used.push_back(false);
    for(int j = 0; j < (int)Random_learning_image.size(); j++){
        srand(time(NULL));
         int R = rand()%96;
         while(R == 18 || R == 19 || im_used[R]) R = rand()%96;
         im_used[R] = true;
         wish_exit.clear();
         wish_exit.resize(94);
         if(R>19) wish_exit[R-2] = 1;
         else wish_exit[R] = 1;
         QString path2 = "/home/beslan/fonts/fonts png/New/" + list.at(R).fileName();
         QDir dir2(path2);
         dir2.setFilter(QDir::Files);
         QFileInfoList list2 = dir2.entryInfoList();
         int R2 = rand()%675;
         Mat image;
         image = imread(list2.at(R2).absoluteFilePath().toStdString(),CV_LOAD_IMAGE_GRAYSCALE);
         Random_learning_image[j] = image;
         Wish_Exit[j] = wish_exit;
         cout<<"end" << j<<endl;
    }
    for(int j = 0; j < (int)Random_learning_image.size(); j++){
        Mat im;
        resize(Random_learning_image[j],im,Size(32,32));
        Random_learning_image[j] = im;
    }
}
/*
void Learning_NN(){
    vector<double> Real_Exit;
    double error = 0;
    for(int i = 0; i < kol_epoch; i++){
        Generated_learning_sequence_image();
        for(int j = 0; j < (int)Random_learning_image.size(); j++){
            if(i == 0 && j ==0) Real_Exit = Forward(matr_in_vect(Random_learning_image[j]),0);
            else Real_Exit = Forward(matr_in_vect(Random_learning_image[j]),1);
            for(int k = 0; k < (int)Real_Exit.size(); k++) {
                error += pow((Wish_Exit[j][k] - Real_Exit[k]),2);
                sum_error_stol_matr_w_layers[2].push_back(Wish_Exit[j][k] - Real_Exit[k]);
            }
            error/=2;
            if(error >= 0.15){

            }
        }
    }
}
*/

int main(int argc, char *argv[])
{
 //   QTextCodec::setCodecForLocale(QTextCodec::codecForName("UTF-8"));
 //   QTextCodec::setCodecForTr(QTextCodec::codecForName("UTF-8"));
 //   QTextCodec::setCodecForCStrings(QTextCodec::codecForName("UTF-8"));
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

//    Mat im2 = imread("/home/beslan/fonts/fonts png/New/1/Alegreya-BlackItalic.png",CV_LOAD_IMAGE_GRAYSCALE);
//    Mat im;
//    resize(im2,im,Size(32,32));
//    imshow("Sdffsd",im);
/*    vector<double> matr_in_vect1 = matr_in_vect(im);
    matr_w_layers[0].resize(matr_in_vect(im).size()/2);
    for(int j = 0; j < (int)matr_w_layers[0].size(); j++) matr_w_layers[0][j].resize(matr_in_vect(im).size());

    matr_w_layers[1].resize(matr_w_layers[0].size()/2);
    for(int j = 0; j < (int)matr_w_layers[1].size(); j++) matr_w_layers[1][j].resize(matr_w_layers[0].size());

    matr_w_layers[2].resize(94);
    for(int j = 0; j < (int)matr_w_layers[2].size(); j++) matr_w_layers[2][j].resize(matr_w_layers[1].size());

    vector<vector<double>> qq = random_matr_w(12,3);
    Connection_Database_MYSQL(qq,0);
    Read_Weights_From_Database();
*/
/*    Generated_learning_sequence_image();
    cout<<"end 1"<<endl;
    vector<double> res1 = Forward(matr_in_vect(Random_learning_image[0]),0);
    cout<<"end 2"<<endl;
//    imshow("Display",Random_learning_image_Wish_exit[0].first);
    for(int i=0;i<(int)res1.size();i++) cout<<res1[i]<<endl;
*/










    //    cout<<"Learning_Rate : ";
//    scanf("%d",Learning_Rate);
//    cin >> Learning_Rate;
/*    String path = "/home/beslan/fonts/fonts png/New/Numbers/0/Alegreya-BlackItalic.png";
    Mat image2 = imread(path,CV_LOAD_IMAGE_GRAYSCALE);
    Mat need_image;
    resize(image2,need_image,Size(32,32));
    imshow("Display",need_image);
    vector<vector<double>> black_white;
    black_white.resize(need_image.rows);
    for (int i = 0; i < black_white.size(); i++) black_white[i].resize(need_image.rows);
    for (int i = 0; i < need_image.rows; i++)
        for (int j = 0; j < need_image.cols; j++)
            if((int)(need_image.at<Vec3b>(i, j)[0]/255) != 0)
                black_white[i][j] = (need_image.at<Vec3b>(i, j)[0]/255);
    vector<double> vect_image = matr_in_vect(black_white);*/
//    Generated_learning_sequence_image();
/*
    for(int i = 0; i < (int)Random_learning_image.size(); i++){
//        Mat image4 = imread(random_learning_image[i],CV_LOAD_IMAGE_GRAYSCALE);
//        imshow("Display  " + i,image4);
        imshow("Display  " + i,Random_learning_image[i]);
    }*/
/*    vector<double> res_forward = Forward(vect_image,0);
    double error = 0;
    for(int i=0;i<res_forward.size();i++) error += sqr(wish_exit[i] - res_forward[i]);
    error/=2;*/
    return a.exec();
}








