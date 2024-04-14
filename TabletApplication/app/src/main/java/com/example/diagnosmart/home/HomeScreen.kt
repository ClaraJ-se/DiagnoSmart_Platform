package com.example.diagnosmart.home

import android.util.Log
import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import com.example.diagnosmart.data.source.models.Patient
import com.example.symtomsavvy.R

@Composable
fun HomeScreen(
        onBack: () -> Unit,
        onAddPatient: () -> Unit,
        onSettingOpen: () -> Unit,
        onLoginError: () -> Unit,
        viewModel: HomeViewModel = hiltViewModel()
) {

    val uiState by viewModel.uiState.collectAsState()
    val context = LocalContext.current
    val errorText = stringResource(id = R.string.error_common)

//    BackHandler(onBack = { Timber.d("Go back is disabled") })

    Box(
            modifier = Modifier
                .fillMaxSize()
    ) {


        Row(modifier = Modifier.fillMaxSize()) {
            Column(
                    modifier = Modifier
                        .weight(1f)
                        .background(colorResource(id = R.color.color_background)),
                    verticalArrangement = Arrangement.Top,
                    horizontalAlignment = Alignment.CenterHorizontally

            ) {

                AddImageButton(modifier = Modifier.padding(24.dp)) { }

                val patientList = uiState.patients

                LazyVerticalGrid(
                        columns = GridCells.Fixed(2),
                        content = {
                            items(patientList.size) { patient ->

                                ItemPatient(
                                        modifier = Modifier
                                            .width(100.dp)
                                            .height(150.dp)
                                            .padding(24.dp),
                                        backgroundColor = Color.White,
                                        cornerRadius = 8.dp,
                                        patientList[patient]
                                )

                            }
                        },
                        modifier = Modifier.weight(1f)
                )

            }
        }
    }


    LaunchedEffect(uiState) {
        if (uiState.isTaskCompleted) {
            onAddPatient()
        } else if (uiState.error != null) {
            Toast.makeText(context, errorText + "\n" + uiState.error, Toast.LENGTH_LONG).show()
        }
    }
}


@Composable
fun ItemPatient(
        modifier: Modifier = Modifier,
        backgroundColor: Color,
        cornerRadius: Dp = 8.dp,
        patient: Patient
) {
    Box(
            modifier = modifier
                .background(color = backgroundColor, shape = RoundedCornerShape(cornerRadius))
    ) {
        Row {
            Column(modifier = Modifier.weight(0.5f)) {

            }
            Column(modifier = Modifier.weight(1f)) {
                Text(
                        text = "%s %s".format(patient.firstName, patient.lastName),
                        modifier = Modifier
                            .padding(vertical = 8.dp, horizontal = 16.dp)
                            .wrapContentSize(),
                        textAlign = TextAlign.Center,
                        fontSize = 20.sp
                )
                Text(
                        text = "Room number: %s".format(patient.roomNumber),
                        modifier = Modifier
                            .padding(vertical = 8.dp, horizontal = 16.dp)
                            .wrapContentSize(),
                        textAlign = TextAlign.Center,
                        fontSize = 20.sp
                )
            }
        }

    }
}

@Composable
fun AddImageButton(modifier: Modifier,
                   onClick: () -> Unit) {
    Box(
            modifier = Modifier
                .background(
                        color = colorResource(id = R.color.color_3),
                        shape = RoundedCornerShape(8.dp)
                )
                .width(600.dp)
                .height(150.dp)
                .clickable { onClick() }, contentAlignment = Alignment.Center
    ) {

        Text(
                text = stringResource(id = R.string.add_patient),
                fontSize = 35.sp,
                modifier = Modifier
                    .padding(vertical = 8.dp, horizontal = 16.dp)
                    .wrapContentSize(),
                textAlign = TextAlign.Center

        )

    }
}


