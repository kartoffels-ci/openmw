#include "gpustate.hpp"

#include <chrono>
#include <iomanip>
#include <limits>

#include <osg/State>
#include <osg/Image>
#include <osg/Texture2D>

#include <components/state/resourcemanager.hpp>

#include <components/debug/debuglog.hpp>
#include <components/settings/values.hpp>

#include "texture.hpp"

namespace State
{
    GPUState::GPUState(std::shared_ptr<ResourceManager> resourceManager)
        : osg::Drawable()
        , mResourceManager(std::move(resourceManager))
    {
        getOrCreateStateSet()->setRenderBinDetails(-1, "RenderBin");
        setCullingActive(false);
        setUpdateCallback(new osg::NodeCallback);
    }

    void GPUState::makeGPUHandle(osg::State& state, BindlessTextureHandle& handle)
    {
        // Needs to map to the contextId, textures are resident per context
        if (handle.handle.has_value())
            return;

        ++mUploadCount;

        const unsigned int contextId = state.getContextID();

        auto* ext = state.get<osg::GLExtensions>();
        handle.texture->apply(state);
        state.applyTextureAttribute(0, handle.texture);

        osg::Texture::TextureObject* textureObject = handle.texture->getTextureObject(contextId);
        if (!textureObject)
            return;

        // GL spec: texture parameters (filter, wrap) become immutable after this call.
        // The ShaderVisitor strips bindless textures from StateSets to prevent OSG from
        // re-applying parameters. If any code path modifies this texture's GL params
        // after this point, AMD will crash.
        handle.handle = ext->glGetTextureHandle(textureObject->id());
    }

    int GPUState::open(osg::State& state, BindlessTextureHandle& handle)
    {
        // Needs to map to the contextId, textures are resident per context
        if (handle.resident)
        {
            mTextures[handle.virtualIndex] = *handle.handle;
            return handle.virtualIndex;
        }

        // Ensure the handle exists
        makeGPUHandle(state, handle);

        if (!handle.handle.has_value() || *handle.handle == 0)
        {
            Log(Debug::Error) << "Failed to create bindless handle for texture index " << handle.virtualIndex
                              << " (" << handle.filename << ")";
            mTextures[handle.virtualIndex] = mFallbackHandle;
            return handle.virtualIndex;
        }

        auto* ext = state.get<osg::GLExtensions>();
        ext->glMakeTextureHandleResident(*handle.handle);
        handle.resident = true;
        mTextures[handle.virtualIndex] = *handle.handle;

        // Log(Debug::Error) << "[RESIDENT] " << handle.handle.value() << " @VINDEX => " << handle.virtualIndex;
        return handle.virtualIndex;
    }

    void GPUState::close(osg::State& state, BindlessTextureHandle& handle)
    {
        if (!handle.resident)
            return;

        auto* ext = state.get<osg::GLExtensions>();
        ext->glMakeTextureHandleNonResident(*handle.handle);
        handle.resident = false;
    }

    void GPUState::initFallbackTexture(osg::State& state)
    {
        osg::ref_ptr<osg::Image> image = new osg::Image;
        image->allocateImage(1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE);
        unsigned char* data = image->data();
        data[0] = 255; data[1] = 255; data[2] = 255; data[3] = 255;

        mFallbackTexture = new osg::Texture2D(image);
        mFallbackTexture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::NEAREST);
        mFallbackTexture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::NEAREST);
        mFallbackTexture->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
        mFallbackTexture->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);

        mFallbackTexture->apply(state);
        state.applyTextureAttribute(0, mFallbackTexture);

        const unsigned int contextId = state.getContextID();
        osg::Texture::TextureObject* textureObject = mFallbackTexture->getTextureObject(contextId);
        if (!textureObject)
        {
            Log(Debug::Error) << "Failed to create fallback bindless texture";
            return;
        }

        auto* ext = state.get<osg::GLExtensions>();
        mFallbackHandle = ext->glGetTextureHandle(textureObject->id());
        if (mFallbackHandle == 0)
        {
            Log(Debug::Error) << "glGetTextureHandle returned 0 for fallback texture";
            return;
        }

        ext->glMakeTextureHandleResident(mFallbackHandle);
        Log(Debug::Info) << "Bindless fallback texture ready, handle=" << mFallbackHandle;
    }

    void GPUState::drawImplementation(osg::RenderInfo& renderInfo) const
    {
        auto drawStart = std::chrono::high_resolution_clock::now();

        osg::State& state = *renderInfo.getState();
        const osg::GLExtensions* ext = state.get<osg::GLExtensions>();
        const unsigned int frameId = state.getFrameStamp()->getFrameNumber() % 2;

        auto& materials = mQueuedGPUMaterials[frameId];

        if (!mInitialized)
        {
            mInitialized = true;

            ext->glGenBuffers(1, &mSSBOMaterials);
            ext->glGenBuffers(1, &mSSBOTextures);

            const_cast<GPUState*>(this)->initFallbackTexture(state);
        }

        if (!materials.empty())
        {
            mMaterials.resize(std::max(mMaterials.size(), materials.rbegin()->first + 1));

            for (const auto& [index, material] : materials)
            {
                mMaterials[index] = material;
            }

            ext->glBindBuffer(GL_SHADER_STORAGE_BUFFER, mSSBOMaterials);

            size_t newSize = sizeof(Material::GPUData) * mMaterials.size();

            // Orphan-before-write: glBufferData(null) discards the old allocation (GPU
            // keeps reading it) and gives us a fresh one, avoiding GPU/CPU sync stalls.
            // AMD validates SSBO contents at draw time — writing in-place while the GPU
            // reads the previous frame's data is undefined behavior on AMD.
            ext->glBufferData(GL_SHADER_STORAGE_BUFFER, newSize, nullptr, GL_DYNAMIC_DRAW);
            mBufferSize = newSize;

            ext->glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, newSize, mMaterials.data());

            ext->glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            materials.clear();
        }

        ext->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mSSBOMaterials);
        ext->glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        // Merge this frame's queued textures into the persistent pending queue
        auto& textures = mQueuedGPUTextures[frameId];
        if (!textures.empty())
        {
            mPendingTextures.insert(std::make_move_iterator(textures.begin()),
                std::make_move_iterator(textures.end()));
            textures.clear();
        }

        mUploadCount = 0;

        if (!mPendingTextures.empty())
        {
            size_t requiredSize = mPendingTextures.rbegin()->first + 1;
            if (requiredSize > mTextures.size())
                mTextures.resize(requiredSize, mFallbackHandle);

            const bool bulkLoad = mPendingTextures.size() > 32;
            const size_t maxUploadsPerFrame = bulkLoad ? std::numeric_limits<size_t>::max() : 8;
            size_t uploads = 0;

            for (auto it = mPendingTextures.begin(); it != mPendingTextures.end(); )
            {
                auto& [index, handle] = *it;
                bool needsUpload = !handle.handle.has_value();

                if (needsUpload && uploads >= maxUploadsPerFrame)
                {
                    ++it;
                    continue;
                }

                if (handle.filename.empty())
                    Log(Debug::Error) << "Empty filename, this texture doesn't look right " << index;

                const_cast<GPUState*>(this)->open(state, handle);

                if (needsUpload)
                    ++uploads;

                it = mPendingTextures.erase(it);
            }

            if (mTextures.size() > 8192 && mTextures.size() % 1024 == 0)
                Log(Debug::Warning) << "Bindless texture count high: " << mTextures.size()
                                    << " — AMD descriptor heap exhaustion possible";

            ext->glBindBuffer(GL_SHADER_STORAGE_BUFFER, mSSBOTextures);

            size_t newSize = sizeof(std::uint64_t) * mTextures.size();

            // Orphan-before-write (see materials SSBO comment above).
            ext->glBufferData(GL_SHADER_STORAGE_BUFFER, newSize, nullptr, GL_DYNAMIC_DRAW);
            mTextureBufferSize = newSize;

            ext->glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, newSize, mTextures.data());

            ext->glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        ext->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mSSBOTextures);
        ext->glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        // makeGPUHandle pollutes texture unit 0 via apply()/applyTextureAttribute().
        // Dirty all attributes so OSG re-applies state for the next drawable instead
        // of trusting its stale tracking. Without this, AMD may skip a required texture
        // rebind and render with the wrong texture or crash.
        if (mUploadCount > 0)
        {
            state.dirtyAllAttributes();
            state.dirtyAllModes();
        }

        auto drawEnd = std::chrono::high_resolution_clock::now();
        double drawMs = std::chrono::duration<double, std::milli>(drawEnd - drawStart).count();

        if (Settings::general().mFrameProfiling && (mUploadCount > 0 || drawMs > 5.0))
            Log(Debug::Error) << "[GPUState] frame " << state.getFrameStamp()->getFrameNumber()
                              << " uploads=" << mUploadCount
                              << " pending=" << mPendingTextures.size()
                              << " drawImpl=" << std::fixed << std::setprecision(1) << drawMs << "ms";
    }

    void GPUState::releaseGLObjects(osg::State* state) const
    {
        if (!state)
        {
            osg::Drawable::releaseGLObjects(state);
            return;
        }

        auto* ext = state->get<osg::GLExtensions>();

        // Make all texture handles non-resident
        for (auto& handle : mTextures)
        {
            if (handle != 0 && handle != mFallbackHandle)
                ext->glMakeTextureHandleNonResident(handle);
        }
        mTextures.clear();

        // Clean up any pending handles that were made resident
        for (auto& [idx, handle] : mPendingTextures)
        {
            if (handle.resident && handle.handle.has_value())
                ext->glMakeTextureHandleNonResident(*handle.handle);
        }
        mPendingTextures.clear();

        if (mFallbackHandle != 0)
        {
            ext->glMakeTextureHandleNonResident(mFallbackHandle);
            mFallbackHandle = 0;
        }

        // Delete SSBOs
        if (mSSBOMaterials)
        {
            ext->glDeleteBuffers(1, &mSSBOMaterials);
            mSSBOMaterials = 0;
        }
        if (mSSBOTextures)
        {
            ext->glDeleteBuffers(1, &mSSBOTextures);
            mSSBOTextures = 0;
        }

        mBufferSize = 0;
        mTextureBufferSize = 0;
        mInitialized = false;

        osg::Drawable::releaseGLObjects(state);
    }

    void GPUState::traverse(osg::NodeVisitor& nv)
    {
        if (nv.getVisitorType() == osg::NodeVisitor::UPDATE_VISITOR)
        {
            size_t frameId = nv.getFrameStamp()->getFrameNumber() % 2;
            auto drained = mResourceManager->drainQueues();
            mQueuedGPUMaterials[frameId] = std::move(drained.materials);
            mQueuedGPUTextures[frameId] = std::move(drained.textures);
        }
    }
}