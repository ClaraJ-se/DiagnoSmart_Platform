package com.example.diagnosmart.login

import android.widget.Toast
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.VisibilityOff
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import com.example.symtomsavvy.R

@Composable
fun LoginScreen(
        onBack: () -> Unit,
        onLoginComplete: () -> Unit,
        openAlternativeLogin: () -> Unit,
        onSettingOpen: () -> Unit,
        onLoginError: () -> Unit,
        viewModel: LoginViewModel = hiltViewModel()
) {

    val uiState by viewModel.uiState.collectAsState()
    val context = LocalContext.current
    val errorText = stringResource(id = R.string.error_common)

    Box(
            modifier = Modifier
                .background(colorResource(id = R.color.color_1))
                .fillMaxSize()
    ) {

        Column(
                modifier = Modifier
                    .fillMaxSize(),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
        ) {
            Column(
                    modifier = Modifier
                        .width(400.dp)
                        .wrapContentHeight()
                        .weight(1f)
                        .verticalScroll(rememberScrollState()),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
            ) {

                Icon(
                        painter = painterResource(id = R.drawable.profile_ic),
                        contentDescription = null,
                        modifier = Modifier
                            .padding(start = 8.dp),
                        tint = colorResource(id = R.color.color_2)
                )


                Text(
                        modifier = Modifier
                            .padding(top = 16.dp)
                            .align(Alignment.Start),
                        text = stringResource(id = R.string.username)
                )

                EmailInputField(
                        modifier = Modifier
                            .width(400.dp)
                            .wrapContentHeight(),
                        viewModel = viewModel
                )

                Text(
                        modifier = Modifier
                            .padding(top = 16.dp)
                            .align(Alignment.Start),
                        text = stringResource(id = R.string.password),

                        )

                PasswordInputField(
                        modifier = Modifier
                            .width(400.dp)
                            .wrapContentHeight(),
                        viewModel = viewModel
                )

                LoginContinueButton(
                        modifier = Modifier
                            .width(400.dp)
                            .height(80.dp)
                            .padding(top = 16.dp),
                        handleLogin = { viewModel.handleLogin() }
                )

                Icon(
                        painter = painterResource(id = R.drawable.diagnosmart_platform_text),
                        contentDescription = null,
                        modifier = Modifier
                            .padding(start = 16.dp, top = 16.dp, end = 16.dp, bottom = 8.dp),
                        tint = colorResource(id = R.color.color_2)
                )

                Icon(
                        painter = painterResource(id = R.drawable.path_to_unbiased_diagnosis),
                        contentDescription = null,
                        modifier = Modifier.padding(
                                start = 32.dp, top = 8.dp, end = 32.dp, bottom = 8.dp
                        ),
                        tint = colorResource(id = R.color.color_2)
                )

            }
        }

        if (uiState.isLoading) {
            Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .clickable {},
                    contentAlignment = Alignment.Center,
            ) {
                CircularProgressIndicator(
                        modifier = Modifier.size(128.dp),
                        color = Color.LightGray,
                        strokeWidth = 10.dp
                )
            }

        }
    }

    LaunchedEffect(uiState) {
        if (uiState.isTaskCompleted) {
            onLoginComplete()
        } else if (uiState.error != null) {
            Toast.makeText(context, errorText + "\n" + uiState.error, Toast.LENGTH_LONG).show()
        }
    }

}

@Composable
fun EmailInputField(
        modifier: Modifier = Modifier,
        viewModel: LoginViewModel
) {

    OutlinedTextField(
            modifier = modifier,
            value = viewModel.email,
            onValueChange = { viewModel.updateEmail(it) },
            keyboardOptions = KeyboardOptions(
                    keyboardType = KeyboardType.Email, imeAction = ImeAction.Done
            ),
            label = { Text(text = "") }
    )
}


@Composable
fun PasswordInputField(
        modifier: Modifier = Modifier,
        viewModel: LoginViewModel
) {

    var showPassword by rememberSaveable { mutableStateOf(false) }

    OutlinedTextField(
            modifier = modifier,
            value = viewModel.password,
            onValueChange = { viewModel.updatePassword(it) },
            visualTransformation = if (showPassword) VisualTransformation.None else PasswordVisualTransformation(),
            keyboardOptions = KeyboardOptions(
                    keyboardType = KeyboardType.Password, imeAction = ImeAction.Done
            ),
            label = { Text(text = "") },
            trailingIcon = {
                val image = if (showPassword)
                    Icons.Filled.Visibility
                else Icons.Filled.VisibilityOff

                val description =
                    if (showPassword) stringResource(id = R.string.test) else stringResource(
                            id = R.string.error_common
                    )
                IconButton(onClick = { showPassword = !showPassword }) {
                    Icon(imageVector = image, description)
                }
            }
    )
}

@Composable
fun LoginContinueButton(
        modifier: Modifier,
        handleLogin: () -> Unit
) {

    Button(
            modifier = modifier,
            colors = ButtonDefaults.buttonColors(
                    containerColor = colorResource(id = R.color.color_2),
                    contentColor = colorResource(id = R.color.color_1)
            ),
            onClick = { handleLogin() },
    ) {
        Text(
                text = stringResource(id = R.string.login_sign_in),
                fontSize = 18.sp,
//                style = typography.button
        )
    }
}

