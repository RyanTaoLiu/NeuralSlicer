void MainWindow::on_pushButtonXXX_Clicked(){
    QString heightFile = QFileDialog::getOpenFileName(this,
                                 QString("Get File of height field"),
                                 QString("."),
                                 QString("*.txt"));
    if( !heightFile.isNull() )
    {
        qDebug() << "selected file path : " << heightFile.toUtf8();
    }

    QFile file(heightFile);
    if(!file.open(QIODevice::ReadOnly)) {
        QMessageBox::information(0, "error", file.errorString());
    }

    QTextStream in(&file);
    QVector<QVector3D> defromedNodes;
    float maxHeight = -1e9;
    float minHeight = 1e9;
    while(!in.atEnd()) {
        QString line = in.readLine();
        QStringList lineList = line.split(' ');
        //float value = line.toFloat();
        defromedNodes.append(QVector3D(lineList[0].toFloat(),lineList[1].toFloat(),lineList[2].toFloat()));
        float y = lineList[1].toFloat();
        if(y < minHeight){minHeight = y;}
        if(y > maxHeight){maxHeight = y;}
    }



    PolygenMesh* tet_Model = this->_detectPolygenMesh(TET_MODEL);
    if (tet_Model == NULL) {
        std::cerr << "There is no Tet mesh, please check." << std::endl; return;
    }
    QMeshPatch* tetMesh = (QMeshPatch*)tet_Model->GetMeshList().GetHead();

    if (S3_case == SUPPORT_LESS
        || S3_case == STRENGTH_REINFORCEMENT
        || (S3_case == SURFACE_QUALITY
            && (tet_Model->getModelName() != "AnkleBaseV1")
            && (S3_case == SURFACE_QUALITY && tet_Model->getModelName() != "curvedprinting")
            && (S3_case == SURFACE_QUALITY && tet_Model->getModelName() != "wedge")
        )
        //|| (S3_case == SURFACE_QUALITY)
        || S3_case == HYBRID_SL_SQ
        || S3_case == HYBRID_SR_SQ
        || S3_case == HYBRID_SL_SR
        || S3_case == HYBRID_SL_SR_SQ
            ) {

        // Planar printing button
        bool planar_printing_Button = false;
        // resume the original position
        if (planar_printing_Button == true) {
            for (GLKPOSITION pos = tetMesh->GetNodeList().GetHeadPosition(); pos != nullptr;) {
                QMeshNode* node = (QMeshNode*)tetMesh->GetNodeList().GetNext(pos);

                double pp[3];
                for (int i = 0; i < 3; i++) { pp[i] = node->initial_coord3D[i]; }
                node->SetCoord3D(pp[0], pp[1], pp[2]);
            }
        }

        // Get height field (scalar field)
        float heightRange = maxHeight - minHeight;
        int nodeIdx = 0;
        for (GLKPOSITION pos = tetMesh->GetNodeList().GetHeadPosition(); pos != nullptr;) {
            QMeshNode* node = (QMeshNode*)tetMesh->GetNodeList().GetNext(pos);
            double xx, yy, zz;
            //node->GetCoord3D(xx, yy, zz);
            QVector3D tempNode = defromedNodes[nodeIdx];
            xx = tempNode[0];
            yy = tempNode[1];
            zz = tempNode[2];
            node->deformed_coord3D << xx, yy, zz;
            node->scalarField_init = yy;
            node->scalarField = (yy - minHeight) / heightRange;
            nodeIdx++;
        }

        // resume the original position
        if (planar_printing_Button == false) {
            for (GLKPOSITION pos = tetMesh->GetNodeList().GetHeadPosition(); pos != nullptr;) {
                QMeshNode* node = (QMeshNode*)tetMesh->GetNodeList().GetNext(pos);

                double pp[3];
                //for (int i = 0; i < 3; i++) { pp[i] = node->initial_coord3D[i]; }
                node->GetCoord3D(pp[0], pp[1], pp[2]);
                node->SetCoord3D(pp[0], pp[1], pp[2]);
            }
        }

        // get a smoothed scalar field (for further support generation)
        this->_scalarField_2_vectorField(true);
        //if (tet_Model->getModelName() != "AnkleBaseV1") {
        if (S3_case == SUPPORT_LESS
            || S3_case == STRENGTH_REINFORCEMENT
            || S3_case == HYBRID_SL_SR
            || (S3_case == SURFACE_QUALITY && tet_Model->getModelName() != "AnkleBaseV1")
            //|| S3_case == HYBRID_SL_SQ
            || (S3_case == HYBRID_SL_SR_SQ && planar_printing_Button)
                ) {
            this->_vectorField_smooth(10, true);
            this->_vectorField_2_scalarField(false);
        }

        std::cout << "Use height field" << std::endl;

    }
    else if ((S3_case == SURFACE_QUALITY) && (tet_Model->getModelName() == "AnkleBaseV1")
             || (S3_case == SURFACE_QUALITY) && (tet_Model->getModelName() == "curvedprinting")
             || (S3_case == SURFACE_QUALITY) && (tet_Model->getModelName() == "wedge")
            ){

        // Get height field (scalar field)
        double boundingBox[6]; double heightRange = 0.0;
        tetMesh->ComputeBoundingBox(boundingBox);
        heightRange = boundingBox[3] - boundingBox[2];
        for (GLKPOSITION pos = tetMesh->GetNodeList().GetHeadPosition(); pos != nullptr;) {
            QMeshNode* node = (QMeshNode*)tetMesh->GetNodeList().GetNext(pos);

            double xx, yy, zz;
            node->GetCoord3D(xx, yy, zz);
            node->deformed_coord3D << xx, yy, zz;
            node->scalarField_init = yy;
            node->scalarField = (yy - boundingBox[2]) / heightRange;
        }

        // resume the original position
        for (GLKPOSITION pos = tetMesh->GetNodeList().GetHeadPosition(); pos != nullptr;) {
            QMeshNode* node = (QMeshNode*)tetMesh->GetNodeList().GetNext(pos);

            double pp[3];
            for (int i = 0; i < 3; i++) { pp[i] = node->initial_coord3D[i]; }
            node->SetCoord3D(pp[0], pp[1], pp[2]);
        }
        this->_compTetMeshVolumeMatrix(tetMesh);

        this->_scalarField_2_vectorField_unNormalized(tetMesh);
        this->_vectorField_2_scalarField(false);//false->local vector // get real scalar field
        // collect the node set of each critical region
        int splitTime = this->_mark_SurfaceKeep_region(tetMesh);
        // calculate the AVERAGE(MAX) scalar value of each node set and foce it to each node
        this->_fixScalarField_surfaceKeepRegion(tetMesh, splitTime);
        // compute scalar field with hard Constrain
        this->_vectorField_2_scalarField_withHardConstrain(tetMesh);
        this->_compScalarField_normalizedValue(tetMesh);
        // update unnormalized vector field
        this->_scalarField_2_vectorField_unNormalized(tetMesh);
        // re-compute scalar field with hard Constrain
        this->_vectorField_2_scalarField_withHardConstrain(tetMesh);
        this->_compScalarField_normalizedValue(tetMesh);


        std::cout << "Use post-process field" << std::endl;
    }
    else {
        // Planar printing button
        bool planar_printing_Button = false;
        // resume the original position
        if (planar_printing_Button == true) {
            for (GLKPOSITION pos = tetMesh->GetNodeList().GetHeadPosition(); pos != nullptr;) {
                QMeshNode* node = (QMeshNode*)tetMesh->GetNodeList().GetNext(pos);

                double pp[3];
                for (int i = 0; i < 3; i++) { pp[i] = node->initial_coord3D[i]; }
                node->SetCoord3D(pp[0], pp[1], pp[2]);
            }
        }

        // Get height field (scalar field)
        float heightRange = maxHeight - minHeight;
        int nodeIdx = 0;
        for (GLKPOSITION pos = tetMesh->GetNodeList().GetHeadPosition(); pos != nullptr;) {
            QMeshNode* node = (QMeshNode*)tetMesh->GetNodeList().GetNext(pos);
            double xx, yy, zz;
            //node->GetCoord3D(xx, yy, zz);
            QVector3D tempNode = defromedNodes[nodeIdx];
            xx = tempNode[0];
            yy = tempNode[1];
            zz = tempNode[2];
            node->deformed_coord3D << xx, yy, zz;
            node->scalarField_init = yy;
            node->scalarField = (yy - minHeight) / heightRange;
            nodeIdx++;
        }

        // resume the original position
        if (planar_printing_Button == false) {
            for (GLKPOSITION pos = tetMesh->GetNodeList().GetHeadPosition(); pos != nullptr;) {
                QMeshNode* node = (QMeshNode*)tetMesh->GetNodeList().GetNext(pos);

                double pp[3];
                //for (int i = 0; i < 3; i++) { pp[i] = node->initial_coord3D[i]; }
                node->GetCoord3D(pp[0], pp[1], pp[2]);
                node->SetCoord3D(pp[0], pp[1], pp[2]);
            }
        }

        // get a smoothed scalar field (for further support generation)
        this->_scalarField_2_vectorField(true);
        //if (tet_Model->getModelName() != "AnkleBaseV1") {
        if (S3_case == SUPPORT_LESS
            || S3_case == STRENGTH_REINFORCEMENT
            || S3_case == HYBRID_SL_SR
            || (S3_case == SURFACE_QUALITY && tet_Model->getModelName() != "AnkleBaseV1")
            //|| S3_case == HYBRID_SL_SQ
            || (S3_case == HYBRID_SL_SR_SQ && planar_printing_Button)
                ) {
            this->_vectorField_smooth(10, true);
            this->_vectorField_2_scalarField(false);
        }

        std::cout << "Use height field" << std::endl;
        //std::cout << "this is unknown S3_case" << std::endl;
    }

    // supplementary code
    if (tet_Model->getModelName() != "AnkleBaseV1") {
        ui->pushButton_isoLayerGeneration->setEnabled(true);
    }
    else {
        ui->pushButton_adaptiveHeightSlicing->setEnabled(true);
        ui->pushButton_isoLayerGeneration->setEnabled(true);
        //ui->checkBox_outputLayer->setChecked(true);
    }
    //ui->pushButton_isoLayerGeneration->setEnabled(true);
    //ui->pushButton_output_QvectorField->setEnabled(true);
    //ui->pushButton_output_ScalarOrHeight_Field->setEnabled(true);
    tetMesh->drawStressField = false;
    std::cout << "\nFinish inverse deformation. " << std::endl;
    pGLK->refresh(true);
    pGLK->Zoom_All_in_View();
}
