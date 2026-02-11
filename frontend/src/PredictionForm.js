// src/PredictionForm.js
import React, { useState } from 'react';
import axios from 'axios';
import './PredictionForm.css'; // For styling

function PredictionForm() {
    const [selectedImage, setSelectedImage] = useState(null);
    const [previewImage, setPreviewImage] = useState(null);
    const [age, setAge] = useState('');
    const [sex, setSex] = useState(''); // e.g., 'male', 'female', 'other'
    const [localization, setLocalization] = useState(''); // e.g., 'back', 'face'
    // Add more state variables for other tabular inputs your model needs
    // Example: const [dxType, setDxType] = useState('');

    const [predictionResult, setPredictionResult] = useState(null);
    const [confidence, setConfidence] = useState(null); // AI’dan gelen tahmin sonucu
    const [risk_level, setRiskLevel] = useState(null)
    const [recommendation, setRecommendation] = useState(null)
    const [isLoading, setIsLoading] = useState(false);  // Şu anda sunucuya istek gidiyor mu, bekleniyor mu?
    const [error, setError] = useState('');

    const handleImageChange = (event) => {
        const file = event.target.files[0];  //file: Artık bu, kullanıcının seçtiği resmi temsil eden bir dosya objesi.
        if (file) {  //Eğer gerçekten bir dosya seçildiyse (yani kullanıcı boş geçmediyse) bu blok çalışır.
            setSelectedImage(file);  //selectedImage değişkenine bu dosyayı kaydeder.
            setPreviewImage(URL.createObjectURL(file)); //URL.createObjectURL(file): Tarayıcı içinde bu dosya için geçici bir görsel linki oluşturur. Böylece bu dosya hemen ekranda önizleme olarak gösterilebilir (sunucuya yüklemeden önce bile!).
            setPredictionResult(null); // Reset previous prediction
            setConfidence(null);
            setError(''); //Daha önce bir hata mesajı gösterilmişse onu da temizler.
        }
    };

    const handleSubmit = async (event) => { //Bu kontroller sayesinde backend’e eksik veya hatalı veri gitmesi engellenir.
        event.preventDefault(); //Tarayıcının varsayılan davranışını (örneğin sayfa yenilemeyi) engeller. Normalde <form> gönderilince sayfa yenilenir. Bunu engeller.React'te form gönderirken sayfanın yenilenmesini istemeyiz. Çünkü bu sayfa sıfırlanır, state'ler silinir.
        if (!selectedImage) {
            setError('Please upload an image.');
            return;
        }
        if (!age || !sex || !localization /* Add other required fields */) {
            setError('Please fill in all required fields.');
            return;
        }

        setIsLoading(true);  //"Veri gönderiliyor" gibi düşün. Bu, yükleniyor animasyonu için.
        setError(''); //Önceki hata varsa temizle.
        setPredictionResult(null); //Eski tahmin varsa onları da sıfırla. Yeni sonuç için sayfa temizlenmiş oluyor.
        setConfidence(null);

        const formData = new FormData();
        formData.append('image', selectedImage); // Key 'image' must match backend expectation

        // Append tabular data. Backend might expect this as separate fields or as a JSON string
        // Option 1: Separate fields (if backend handles them individually)
        formData.append('age', age);
        formData.append('sex', sex);
        formData.append('localization', localization);
        // formData.append('dx_type', dxType);


        // Option 2: As a JSON string (often cleaner for many fields)
        // const tabularData = {
        //     age: parseInt(age), // Ensure correct data types
        //     sex: sex,
        //     localization: localization,
        //     // dx_type: dxType,
        //     // ... other fields
        // };
        // formData.append('tabular_data', JSON.stringify(tabularData));


        try {
            // Replace with your actual backend API endpoint
            const response = await axios.post('http://localhost:5000/predict', formData, { // axios.post(...) → backend’e bir POST isteği gönderiyor. await → veri gelene kadar bekle. 'http://localhost:5000/predict' → bu, tahmin yapan API’nin adresi.formData → yukarıda kargoya koyduğumuz veriler gönderiliyor.
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setPredictionResult(response.data.prediction); // e.g., "Benign" or "Cancerous" Biz de bunları React state'e kaydederiz ki, ekranda gösterelim.
            setConfidence(response.data.confidence);   // e.g., 0.85  Backend şu bilgileri döner: setConfidence ve setPredictionResult
            setRecommendation(response.data.recommendation);
            setRiskLevel(response.data.risk_level);
        } catch (err) {
            console.error("Prediction API error:", err);
            if (err.response && err.response.data && err.response.data.error) {
                setError(err.response.data.error);
            } else {
                setError('Failed to get prediction. Please try again.');
            }
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="prediction-form-container">
            <h2>Skin Lesion Prediction</h2>
            <form onSubmit={handleSubmit}>  {/*Bu kod bloğu, form gönderildiğinde (yani kullanıcı "Predict" butonuna bastığında) çalışan handleSubmit adlı bir fonksiyondur. */}
                <div className="form-group">
                    <h4 htmlFor="imageUpload">Upload Lesion Image:</h4>
                    <input
                        type="file"
                        id="imageUpload"
                        accept="image/jpeg, image/png"
                        onChange={handleImageChange} /* Bu satırda kullanıcı bir dosya (resim) seçtiğinde onChange olayı tetiklenir. Bu olay gerçekleştiğinde React otomatik olarak event adında bir nesne oluşturur ve onu handleImageChange fonksiyonuna verir. */
                        required
                    />
                    {previewImage && (
                        <div className="image-preview-container">
                            <img src={previewImage} alt="Preview" className="image-preview" />
                        </div>
                    )}
                </div>

                <div className="form-group">
                    <h4 htmlFor="age">Age:</h4>
                    <input
                        type="number"
                        id="age"
                        value={age}
                        onChange={(e) => setAge(e.target.value)}
                        min="0"
                        max="120"
                        required
                    />
                </div>

                <div className="form-group">
                    <h4 htmlFor="sex">Gender:</h4>
                    <select id="sex" value={sex} onChange={(e) => setSex(e.target.value)} required>
                        <option value="">Select Sex</option>
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                        <option value="other">Other</option>
                        {/* Ensure these values match what your model's preprocessor expects */}
                    </select>
                </div>

                <div className="form-group">
                    <h4 htmlFor="localization">Lesion Localization:</h4>
                    <select id="localization" value={localization} onChange={(e) => setLocalization(e.target.value)} required>
                        <option value="">Select Localization</option>
                        <option value="face">Face</option>
                        <option value="back">Back</option>
                        <option value="chest">Chest</option>
                        <option value="upper extremity">Upper Extremity</option>
                        <option value="lower extremity">Lower Extremity</option>
                        <option value="abdomen">Abdomen</option>
                        <option value="trunk">Trunk</option>
                        <option value="foot">Foot</option>
                        <option value="hand">Hand</option>
                        <option value="neck">Neck</option>
                        <option value="scalp">Scalp</option>
                        <option value="ear">Ear</option>
                        <option value="genital">Genital</option>
                        <option value="acral">Acral</option>
                        <option value="unknown">Unknown</option>
                        {/* Populate with all options your 'localization' feature expects */}
                    </select>
                </div>


                <button type="submit" disabled={isLoading}>
                    {isLoading ? 'Predicting...' : 'Predict'}
                </button>
            </form>

            {error && <p className="error-message">{error}</p>}

            {predictionResult && (
                <div className={`prediction-result ${predictionResult.toLowerCase()}`}>
                    <h3>Prediction: {predictionResult}</h3>
                    {confidence !== null && <p>Confidence: {(confidence * 100).toFixed(2)}%</p>}
                    <p>Risk Level: {risk_level}</p>
                    <p>Recommentation: {recommendation}</p>
                    <p className="disclaimer">
                        <strong>Disclaimer:</strong> This is an AI-generated prediction and should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider.
                    </p>
                </div>
            )}
        </div>
    );
}

export default PredictionForm;