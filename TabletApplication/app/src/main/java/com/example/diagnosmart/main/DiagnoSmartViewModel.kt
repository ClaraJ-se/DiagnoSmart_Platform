package com.example.diagnosmart.main

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.diagnosmart.data.source.repo.AccountRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class MainStateTaskUiState(
        val isSplashFinished: Boolean = false,
        val isSessionActive: Boolean = false,

        val deviceId: String? = null,
        val requestPermission: Boolean = true,
)

@HiltViewModel
class DiagnoSmartViewModel @Inject constructor(
        private val accountRepository: AccountRepository,
//        private val settingsRepository: SettingsRepository,
) : ViewModel() {

    private val _mainState = MutableStateFlow(MainStateTaskUiState(isSessionActive = false))
    val mainState: StateFlow<MainStateTaskUiState> = _mainState.asStateFlow()

    init {
        launchSplash()
        checkIfSessionActive()
    }

    private fun launchSplash() {
        viewModelScope.launch {
            delay(3000) // 3000 milliseconds = 3 seconds
            navigateToMainContent()
        }
    }

    private fun navigateToMainContent() {
        _mainState.update {
            it.copy(isSplashFinished = true)
        }

    }


    private fun getDeviceId() {
        viewModelScope.launch {
//            _authState.update {
//                it.copy(
//                        deviceId = settingsRepository.getDeviceId()
//                )
//            }
        }
    }

     fun checkIfSessionActive() {
        viewModelScope.launch {
            _mainState.update {
                it.copy(isSessionActive = accountRepository.isSessionActive())
            }
        }
    }

//    fun dismissDialog() {
//        visiblePermissionDialogQueue.removeFirst()
//    }
//
//    fun onPermissionResult(
//            permission: String,
//            isGranted: Boolean
//    ) {
//        if (!isGranted && !visiblePermissionDialogQueue.contains(permission)) {
//            visiblePermissionDialogQueue.add(permission)
//        }
//    }
//
//    fun requestAppPermission() {
//        viewModelScope.launch {
//            _authState.update {
//                it.copy(requestPermission = true)
//            }
//        }
//    }

}